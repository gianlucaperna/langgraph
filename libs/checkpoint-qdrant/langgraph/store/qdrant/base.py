"""Qdrant-backed BaseStore for LangGraph long-term memory."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Literal, cast

import orjson
from langchain_core.embeddings import Embeddings
from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    Range,
    VectorParams,
)

logger = logging.getLogger(__name__)

# Stable UUID namespace for deterministic point IDs.
_NS_UUID = uuid.UUID("c1d2e3f4-a5b6-7890-abcd-ef0123456789")

# Safe filter key pattern (same as sqlite store).
_FILTER_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_.\-]+$")


def _validate_filter_key(key: str) -> None:
    if not _FILTER_KEY_PATTERN.match(key):
        raise ValueError(
            f"Invalid filter key: '{key}'. Must contain only alphanumeric "
            "characters, underscores, dots, and hyphens."
        )


def _item_pid(namespace: tuple[str, ...], key: str) -> str:
    return str(uuid.uuid5(_NS_UUID, ".".join(namespace) + ":" + key))


def _namespace_to_text(namespace: tuple[str, ...]) -> str:
    return ".".join(namespace)


def _now_ts() -> float:
    return time.time()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_to_datetime(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _group_ops(
    ops: Iterable[Op],
) -> tuple[dict[type, list[tuple[int, Op]]], int]:
    grouped: dict[type, list[tuple[int, Op]]] = defaultdict(list)
    total = 0
    for idx, op in enumerate(ops):
        grouped[type(op)].append((idx, op))
        total += 1
    return grouped, total


def _row_to_item(
    pl: dict[str, Any],
    *,
    loader: Callable[[bytes | str | orjson.Fragment], dict[str, Any]] | None = None,
) -> Item:
    val = pl["value"]
    if loader is not None and isinstance(val, (bytes, str, orjson.Fragment)):
        val = loader(val)
    elif isinstance(val, str):
        val = orjson.loads(val)
    return Item(
        namespace=tuple(pl["namespace"]),
        key=pl["key"],
        value=val,
        created_at=datetime.fromisoformat(pl["created_at"]),
        updated_at=datetime.fromisoformat(pl["updated_at"]),
    )


def _row_to_search_item(
    pl: dict[str, Any],
    score: float | None,
    *,
    loader: Callable[[bytes | str | orjson.Fragment], dict[str, Any]] | None = None,
) -> SearchItem:
    val = pl["value"]
    if loader is not None and isinstance(val, (bytes, str, orjson.Fragment)):
        val = loader(val)
    elif isinstance(val, str):
        val = orjson.loads(val)
    return SearchItem(
        namespace=tuple(pl["namespace"]),
        key=pl["key"],
        value=val,
        created_at=datetime.fromisoformat(pl["created_at"]),
        updated_at=datetime.fromisoformat(pl["updated_at"]),
        score=score,
    )


def _build_filter(
    namespace_prefix: tuple[str, ...] | None,
    filter: dict[str, Any] | None,
    *,
    exclude_expired: bool = True,
) -> Filter | None:
    """Build a Qdrant `Filter` from namespace prefix + payload filter dict."""
    must: list[Any] = []
    must_not: list[Any] = []

    if exclude_expired:
        must_not.append(
            FieldCondition(
                key="expires_at_ts",
                range=Range(lte=_now_ts()),
            )
        )

    if namespace_prefix:
        for i, part in enumerate(namespace_prefix):
            must.append(
                FieldCondition(
                    key=f"namespace[{i}]",
                    match=MatchValue(value=part),
                )
            )

    if filter:
        for key, value in filter.items():
            _validate_filter_key(key)
            if isinstance(value, dict):
                for op_name, val in value.items():
                    must.extend(_filter_op_conditions(f"value.{key}", op_name, val))
            else:
                must.append(
                    FieldCondition(
                        key=f"value.{key}",
                        match=MatchValue(value=value),
                    )
                )

    if not must and not must_not:
        return None
    kwargs: dict[str, Any] = {}
    if must:
        kwargs["must"] = must
    if must_not:
        kwargs["must_not"] = must_not
    return Filter(**kwargs)


def _filter_op_conditions(field: str, op: str, value: Any) -> list[FieldCondition]:
    if op == "$eq":
        return [FieldCondition(key=field, match=MatchValue(value=value))]
    elif op == "$ne":
        # Qdrant handles NOT via must_not; we raise to let the caller handle it.
        raise ValueError("$ne operator must be handled by the caller via must_not")
    elif op == "$gt":
        return [FieldCondition(key=field, range=Range(gt=value))]
    elif op == "$gte":
        return [FieldCondition(key=field, range=Range(gte=value))]
    elif op == "$lt":
        return [FieldCondition(key=field, range=Range(lt=value))]
    elif op == "$lte":
        return [FieldCondition(key=field, range=Range(lte=value))]
    else:
        raise ValueError(f"Unsupported filter operator: {op}")


def _build_filter_with_ne(
    namespace_prefix: tuple[str, ...] | None,
    filter: dict[str, Any] | None,
    *,
    exclude_expired: bool = True,
) -> Filter | None:
    """Extended `_build_filter` that correctly handles `$ne` via `must_not`."""
    must: list[Any] = []
    must_not: list[Any] = []

    if exclude_expired:
        must_not.append(
            FieldCondition(
                key="expires_at_ts",
                range=Range(lte=_now_ts()),
            )
        )

    if namespace_prefix:
        for i, part in enumerate(namespace_prefix):
            must.append(
                FieldCondition(
                    key=f"namespace[{i}]",
                    match=MatchValue(value=part),
                )
            )

    if filter:
        for key, value in filter.items():
            _validate_filter_key(key)
            field = f"value.{key}"
            if isinstance(value, dict):
                for op_name, val in value.items():
                    if op_name == "$ne":
                        must_not.append(
                            FieldCondition(key=field, match=MatchValue(value=val))
                        )
                    else:
                        must.extend(_filter_op_conditions(field, op_name, val))
            else:
                must.append(FieldCondition(key=field, match=MatchValue(value=value)))

    if not must and not must_not:
        return None
    kwargs: dict[str, Any] = {}
    if must:
        kwargs["must"] = must
    if must_not:
        kwargs["must_not"] = must_not
    return Filter(**kwargs)


class QdrantIndexConfig(IndexConfig, total=False):
    """Qdrant-specific index configuration.

    Extends `IndexConfig` with the Qdrant distance metric.
    """

    distance: Literal["cosine", "dot", "euclid", "manhattan"]
    """Vector distance metric.
    - `'cosine'` (default): cosine similarity — best for normalised text embeddings.
    - `'dot'`: dot product.
    - `'euclid'`: Euclidean (L2) distance.
    - `'manhattan'`: Manhattan (L1) distance.
    """


def _distance_from_str(d: str) -> Distance:
    mapping = {
        "cosine": Distance.COSINE,
        "dot": Distance.DOT,
        "euclid": Distance.EUCLID,
        "manhattan": Distance.MANHATTAN,
    }
    if d not in mapping:
        raise ValueError(f"Unknown distance '{d}'. Must be one of: {list(mapping)}")
    return mapping[d]


def _ensure_index_config(
    index: IndexConfig,
) -> tuple[Embeddings, QdrantIndexConfig]:
    """Validate and normalise the index config; return the Embeddings object."""
    if "dims" not in index:
        raise ValueError("IndexConfig must include 'dims'.")
    if "embed" not in index:
        raise ValueError("IndexConfig must include 'embed'.")
    embeddings = ensure_embeddings(index["embed"])
    cfg = cast(QdrantIndexConfig, dict(index))
    cfg.setdefault("distance", "cosine")
    # Pre-tokenise fields for efficient per-put path lookup.
    fields: list[str] = cfg.get("fields") or ["$"]
    cfg["__tokenized_fields"] = [
        (p, "$") if p == "$" else (p, tokenize_path(p)) for p in fields
    ]
    return embeddings, cfg


class BaseQdrantStore:
    """Mixin with all Qdrant store logic (sync-only path)."""

    _client: QdrantClient
    _collection: str
    _index_config: QdrantIndexConfig | None
    _embeddings: Embeddings | None
    _ttl_config: TTLConfig | None
    _executor: concurrent.futures.ThreadPoolExecutor

    def _setup_sync(self) -> None:
        """Create the Qdrant collection and payload indexes (idempotent)."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            if self._index_config:
                dims = int(self._index_config["dims"])
                distance_str = cast(str, self._index_config.get("distance", "cosine"))
                distance = _distance_from_str(distance_str)
                vectors_config: Any = VectorParams(size=dims, distance=distance)
            else:
                vectors_config = {}
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=vectors_config,
            )
        # Payload indexes for fast namespace / key lookups.
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="namespace",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="key",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="expires_at_ts",
            field_schema=PayloadSchemaType.FLOAT,
        )

    # ------------------------------------------------------------------
    # Batch implementation
    # ------------------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        grouped, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        if GetOp in grouped:
            self._handle_get_ops(grouped[GetOp], results)
        if SearchOp in grouped:
            self._handle_search_ops(grouped[SearchOp], results)
        if ListNamespacesOp in grouped:
            self._handle_list_namespaces_ops(grouped[ListNamespacesOp], results)
        if PutOp in grouped:
            self._handle_put_ops(grouped[PutOp])

        return results

    # ------------------------------------------------------------------
    # GetOp
    # ------------------------------------------------------------------

    def _handle_get_ops(
        self,
        ops: list[tuple[int, Op]],
        results: list[Result],
    ) -> None:
        get_ops = cast(list[tuple[int, GetOp]], ops)
        ids = [_item_pid(op.namespace, op.key) for _, op in get_ops]
        pts = self._client.retrieve(
            collection_name=self._collection,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )
        id_to_payload = {str(p.id): p.payload for p in pts}

        now_ts = _now_ts()
        to_refresh: list[str] = []

        for (result_idx, op), pid in zip(get_ops, ids, strict=False):
            pl = id_to_payload.get(pid)
            if pl is None:
                results[result_idx] = None
                continue
            # Item is logically expired — treat as missing.
            expires = pl.get("expires_at_ts")
            if expires is not None and expires <= now_ts:
                results[result_idx] = None
                continue
            results[result_idx] = _row_to_item(pl)
            # Refresh TTL if requested and item has one.
            if op.refresh_ttl and pl.get("ttl_minutes") is not None:
                to_refresh.append(pid)

        if to_refresh:
            for pid in to_refresh:
                pl_ref = id_to_payload[pid]
                if pl_ref is None:
                    continue
                ttl_minutes = pl_ref["ttl_minutes"]
                new_ts = now_ts + ttl_minutes * 60
                self._client.set_payload(
                    collection_name=self._collection,
                    payload={"expires_at_ts": new_ts},
                    points=[pid],
                )

    # ------------------------------------------------------------------
    # PutOp
    # ------------------------------------------------------------------

    def _handle_put_ops(self, ops: list[tuple[int, Op]]) -> None:
        # Deduplicate: last write for each (namespace, key) wins.
        deduped: dict[tuple[tuple[str, ...], str], PutOp] = {}
        for _, op in ops:
            put_op = cast(PutOp, op)
            deduped[(put_op.namespace, put_op.key)] = put_op

        deletes: list[PutOp] = []
        inserts: list[PutOp] = []
        for op in deduped.values():
            (deletes if op.value is None else inserts).append(op)

        # Deletes.
        if deletes:
            delete_ids = [_item_pid(op.namespace, op.key) for op in deletes]
            self._client.delete(
                collection_name=self._collection,
                points_selector=PointIdsList(points=delete_ids),  # type: ignore[arg-type]
            )

        if not inserts:
            return

        # Determine which items need embeddings.
        embed_requests: list[tuple[int, str, str]] = []  # (op_idx, path, text)
        for op_idx, op in enumerate(inserts):
            if op.index is False or self._index_config is None:
                continue
            paths = (
                cast(dict, self._index_config)["__tokenized_fields"]
                if op.index is None
                else [(ix, tokenize_path(ix)) for ix in op.index]
            )
            for path, tokenized_path in paths:
                for text in get_text_at_path(op.value, tokenized_path):
                    embed_requests.append((op_idx, path, text))

        # Compute embeddings in one batch.
        vectors_by_op: dict[int, list[float]] = {}
        if embed_requests and self._embeddings is not None:
            texts = [r[2] for r in embed_requests]
            vecs = self._embeddings.embed_documents(texts)
            # Use the first (or average) vector per op_idx.
            vec_acc: dict[int, list[list[float]]] = defaultdict(list)
            for (op_idx, _, _), vec in zip(embed_requests, vecs, strict=False):
                vec_acc[op_idx].append(vec)
            for op_idx, vec_list in vec_acc.items():
                # Average multiple field vectors.
                dims = len(vec_list[0])
                avg = [sum(v[d] for v in vec_list) / len(vec_list) for d in range(dims)]
                vectors_by_op[op_idx] = avg

        # Fetch existing created_at.
        existing_ids = [_item_pid(op.namespace, op.key) for op in inserts]
        existing_pts = self._client.retrieve(
            collection_name=self._collection,
            ids=existing_ids,
            with_payload=["created_at"],
            with_vectors=False,
        )
        created_at_map = {
            str(p.id): p.payload["created_at"]
            for p in existing_pts
            if p.payload is not None
        }

        now_iso = _now_iso()
        now_ts = _now_ts()
        points: list[PointStruct] = []
        for op_idx, op in enumerate(inserts):
            pid = _item_pid(op.namespace, op.key)
            ttl_minutes = (
                op.ttl
                if op.ttl is not None
                else (self._ttl_config.get("default_ttl") if self._ttl_config else None)
            )
            expires_at_ts: float | None = (
                now_ts + ttl_minutes * 60 if ttl_minutes is not None else None
            )
            payload: dict[str, Any] = {
                "namespace": list(op.namespace),
                "key": op.key,
                # Store as a native dict so Qdrant can index nested fields
                # for payload-filter queries (e.g. value.status == "active").
                # _row_to_item/_row_to_search_item already handle dict values.
                "value": op.value,
                "created_at": created_at_map.get(pid, now_iso),
                "updated_at": now_iso,
                "ttl_minutes": ttl_minutes,
                "expires_at_ts": expires_at_ts,
            }
            vector: Any
            if self._index_config is not None:
                vector = vectors_by_op.get(
                    op_idx, [0.0] * int(self._index_config["dims"])
                )
            else:
                vector = {}
            points.append(PointStruct(id=pid, vector=vector, payload=payload))

        self._client.upsert(collection_name=self._collection, points=points)

    # ------------------------------------------------------------------
    # SearchOp
    # ------------------------------------------------------------------

    def _handle_search_ops(
        self,
        ops: list[tuple[int, Op]],
        results: list[Result],
    ) -> None:
        search_ops = cast(list[tuple[int, SearchOp]], ops)
        # Collect unique query strings to embed in one batch.
        unique_queries: list[str] = list(
            {
                op.query
                for _, op in search_ops
                if op.query and self._index_config is not None
            }
        )
        query_vecs: dict[str, list[float]] = {}
        if unique_queries and self._embeddings is not None:
            vecs = self._embeddings.embed_documents(unique_queries)
            query_vecs = dict(zip(unique_queries, vecs, strict=False))

        for result_idx, op in search_ops:
            filt = _build_filter_with_ne(op.namespace_prefix, op.filter)

            if op.query and op.query in query_vecs and self._index_config:
                # qdrant-client >= 1.10 replaced search() with query_points().
                response = self._client.query_points(
                    collection_name=self._collection,
                    query=query_vecs[op.query],
                    query_filter=filt,
                    limit=op.limit + op.offset,
                    with_payload=True,
                    score_threshold=0.0,
                )
                hits = response.points[op.offset :]
                items: list[SearchItem] = []
                for hit in hits:
                    if hit.payload is None:
                        continue
                    items.append(_row_to_search_item(hit.payload, hit.score))
                results[result_idx] = items
            else:
                all_pts: list[Any] = []
                offset: Any = None
                while True:
                    pts, next_off = self._client.scroll(
                        collection_name=self._collection,
                        scroll_filter=filt,
                        limit=500,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    all_pts.extend(pts)
                    if next_off is None:
                        break
                    offset = next_off
                all_pts.sort(
                    key=lambda p: p.payload.get("updated_at", ""),
                    reverse=True,
                )
                all_pts = all_pts[op.offset : op.offset + op.limit]
                results[result_idx] = [
                    _row_to_search_item(p.payload, None) for p in all_pts
                ]

            # TTL refresh on search.
            if op.refresh_ttl and self._ttl_config:
                self._refresh_ttl_for_results(results[result_idx])

    def _refresh_ttl_for_results(self, items: Any) -> None:
        if not isinstance(items, list):
            return
        now_ts = _now_ts()
        for item in items:
            if not isinstance(item, (Item, SearchItem)):
                continue
            pid = _item_pid(item.namespace, item.key)
            pts = self._client.retrieve(
                collection_name=self._collection,
                ids=[pid],
                with_payload=["ttl_minutes"],
                with_vectors=False,
            )
            if pts and pts[0].payload and pts[0].payload.get("ttl_minutes"):
                ttl = pts[0].payload["ttl_minutes"]
                self._client.set_payload(
                    collection_name=self._collection,
                    payload={"expires_at_ts": now_ts + ttl * 60},
                    points=[pid],
                )

    # ------------------------------------------------------------------
    # ListNamespacesOp
    # ------------------------------------------------------------------

    def _handle_list_namespaces_ops(
        self,
        ops: list[tuple[int, Op]],
        results: list[Result],
    ) -> None:
        for result_idx, op in ops:
            op = cast(ListNamespacesOp, op)
            all_ns: set[tuple[str, ...]] = set()
            offset: Any = None
            while True:
                pts, next_off = self._client.scroll(
                    collection_name=self._collection,
                    scroll_filter=_build_filter_with_ne(None, None),
                    limit=500,
                    offset=offset,
                    with_payload=["namespace"],
                    with_vectors=False,
                )
                for p in pts:
                    if p.payload:
                        all_ns.add(tuple(p.payload["namespace"]))
                if next_off is None:
                    break
                offset = next_off

            filtered = list(all_ns)

            if op.match_conditions:
                for cond in op.match_conditions:
                    path = cond.path
                    if cond.match_type == "prefix":
                        filtered = [ns for ns in filtered if ns[: len(path)] == path]
                    elif cond.match_type == "suffix":
                        filtered = [
                            ns for ns in filtered if ns[len(ns) - len(path) :] == path
                        ]
                    else:
                        logger.warning(
                            "Unknown match_type in list_namespaces: %s",
                            cond.match_type,
                        )

            if op.max_depth is not None:
                filtered = list({ns[: op.max_depth] for ns in filtered})

            filtered.sort()
            results[result_idx] = filtered[op.offset : op.offset + op.limit]

    # ------------------------------------------------------------------
    # TTL sweep
    # ------------------------------------------------------------------

    def sweep_ttl(self) -> int:
        """Delete all expired items. Returns the number of deleted items."""
        from qdrant_client.models import PointIdsList

        expired_pts: list[Any] = []
        offset: Any = None
        filt = Filter(
            must=[
                FieldCondition(
                    key="expires_at_ts",
                    range=Range(lte=_now_ts()),
                )
            ]
        )
        while True:
            pts, next_off = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=filt,
                limit=500,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            expired_pts.extend(pts)
            if next_off is None:
                break
            offset = next_off

        if not expired_pts:
            return 0

        ids = [str(p.id) for p in expired_pts]
        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=ids),  # type: ignore[arg-type]
        )
        return len(ids)


class QdrantStore(BaseStore, BaseQdrantStore):
    """Qdrant-backed store with optional vector similarity search.

    Implements `BaseStore` so it can be passed as ``store=`` to any LangGraph
    agent.  Data persists in a single Qdrant collection.

    !!! example "Basic usage"

        ```python
        from qdrant_client import QdrantClient
        from langgraph.store.qdrant import QdrantStore

        client = QdrantClient(url="http://localhost:6333")
        with QdrantStore.from_client(client) as store:
            store.setup()
            store.put(("users", "alice"), "prefs", {"theme": "dark"})
            item = store.get(("users", "alice"), "prefs")
        ```

    !!! example "With vector search"

        ```python
        from langchain.embeddings import init_embeddings
        from qdrant_client import QdrantClient
        from langgraph.store.qdrant import QdrantStore

        client = QdrantClient(url="http://localhost:6333")
        with QdrantStore.from_client(
            client,
            index={
                "dims": 1536,
                "embed": init_embeddings("openai:text-embedding-3-small"),
                "fields": ["text"],
            },
        ) as store:
            store.setup()
            store.put(("docs",), "doc1", {"text": "Python tutorial"})
            results = store.search(("docs",), query="programming")
        ```

    Args:
        client: An existing `QdrantClient` instance.
        collection_name: Qdrant collection name. Defaults to ``"langgraph_store"``.
        index: Optional `IndexConfig` dict for semantic search.
        ttl: Optional `TTLConfig` dict for item expiration.
    """

    supports_ttl: bool = True

    def __init__(
        self,
        client: QdrantClient,
        *,
        collection_name: str = "langgraph_store",
        index: QdrantIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._collection = collection_name
        self._ttl_config = ttl
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._ttl_thread: threading.Thread | None = None
        self._ttl_stop_event = threading.Event()

        if index is not None:
            self._embeddings, self._index_config = _ensure_index_config(index)
        else:
            self._embeddings = None
            self._index_config = None

    @classmethod
    @contextmanager
    def from_client(
        cls,
        client: QdrantClient,
        *,
        collection_name: str = "langgraph_store",
        index: QdrantIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> Iterator[QdrantStore]:
        """Context-manager factory.

        Args:
            client: A `QdrantClient` instance.
            collection_name: Qdrant collection name.
            index: Optional embedding/index configuration.
            ttl: Optional TTL configuration.

        Yields:
            QdrantStore: A configured store instance.
        """
        store = cls(
            client,
            collection_name=collection_name,
            index=index,
            ttl=ttl,
        )
        try:
            yield store
        finally:
            store._executor.shutdown(wait=False)
            client.close()

    def setup(self) -> None:
        """Create the Qdrant collection and payload indexes (idempotent).

        Must be called once before first use.
        """
        self._setup_sync()

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of store operations synchronously."""
        return BaseQdrantStore.batch(self, ops)

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of store operations asynchronously via thread executor."""
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self.batch, list(ops)
        )

    # ------------------------------------------------------------------
    # TTL sweeper
    # ------------------------------------------------------------------

    def start_ttl_sweeper(self) -> None:
        """Start a background thread that periodically deletes expired items."""
        if self._ttl_config is None:
            return
        interval = self._ttl_config.get("sweep_interval_minutes")
        if interval is None:
            return
        self._ttl_stop_event.clear()
        self._ttl_thread = threading.Thread(
            target=self._ttl_sweep_loop,
            args=(float(interval) * 60,),
            daemon=True,
        )
        self._ttl_thread.start()

    def stop_ttl_sweeper(self) -> None:
        """Signal the TTL sweeper thread to stop and wait for it."""
        self._ttl_stop_event.set()
        if self._ttl_thread is not None:
            self._ttl_thread.join(timeout=10.0)
            self._ttl_thread = None

    def _ttl_sweep_loop(self, interval_secs: float) -> None:
        while not self._ttl_stop_event.wait(timeout=interval_secs):
            try:
                n = self.sweep_ttl()
                if n:
                    logger.debug("QdrantStore TTL sweep: deleted %d items", n)
            except Exception:
                logger.exception("QdrantStore TTL sweep error")

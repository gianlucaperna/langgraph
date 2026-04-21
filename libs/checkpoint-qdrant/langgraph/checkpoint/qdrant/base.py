"""Shared base for Qdrant checkpoint savers."""

from __future__ import annotations

import base64
import random
import threading
import uuid
from collections.abc import Sequence
from typing import Any, cast

import orjson
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_serializable_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

# Stable UUID namespace for deterministic point IDs.
_NS_UUID = uuid.UUID("7f3d2c1a-e5b4-4f8a-9c0d-1e2f3a4b5c6d")

# Single 1-D dummy named vector used for payload-only collections.
_DUMMY_VEC_NAME = "v"
_DUMMY_VEC_CFG: dict[str, Any] = {
    _DUMMY_VEC_NAME: VectorParams(size=1, distance=Distance.DOT)
}
_DUMMY_VEC: dict[str, Any] = {_DUMMY_VEC_NAME: [0.0]}


def _make_id(*parts: str) -> str:
    """Return a deterministic UUID5 string from the given parts."""
    return str(uuid.uuid5(_NS_UUID, ":".join(parts)))


class BaseQdrantSaver(BaseCheckpointSaver[str]):
    """Common logic shared between `QdrantSaver` (sync) and `AsyncQdrantSaver`.

    All state-mutation helpers operate synchronously against a `QdrantClient`.
    The async subclass wraps them with `run_in_executor`.
    """

    _client: QdrantClient
    _prefix: str
    _lock: threading.Lock
    _is_setup: bool

    # ------------------------------------------------------------------
    # Collection name helpers
    # ------------------------------------------------------------------

    @property
    def _chk_collection(self) -> str:
        return f"{self._prefix}checkpoints"

    @property
    def _blob_collection(self) -> str:
        return f"{self._prefix}checkpoint_blobs"

    @property
    def _writes_collection(self) -> str:
        return f"{self._prefix}checkpoint_writes"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        client: QdrantClient,
        *,
        serde: SerializerProtocol | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(serde=serde)
        self._client = client
        self._prefix = prefix
        self._lock = threading.Lock()
        self._is_setup = False

    # ------------------------------------------------------------------
    # Collection bootstrap (idempotent)
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Create Qdrant collections with payload indexes (idempotent).

        Must be called once before first use. Subsequent calls are no-ops.
        """
        with self._lock:
            if self._is_setup:
                return
            self._do_setup()
            self._is_setup = True

    def _do_setup(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        specs: list[tuple[str, list[str]]] = [
            (
                self._chk_collection,
                ["thread_id", "checkpoint_ns", "checkpoint_id"],
            ),
            (
                self._blob_collection,
                ["thread_id", "checkpoint_ns"],
            ),
            (
                self._writes_collection,
                ["thread_id", "checkpoint_ns", "checkpoint_id"],
            ),
        ]
        for name, payload_fields in specs:
            if name not in existing:
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=_DUMMY_VEC_CFG,
                )
            for field in payload_fields:
                self._client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )

    # ------------------------------------------------------------------
    # Deterministic point-ID helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _checkpoint_pid(thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        return _make_id("chk", thread_id, checkpoint_ns, checkpoint_id)

    @staticmethod
    def _blob_pid(
        thread_id: str, checkpoint_ns: str, channel: str, version: str
    ) -> str:
        return _make_id("blob", thread_id, checkpoint_ns, channel, version)

    @staticmethod
    def _write_pid(
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        idx: int,
    ) -> str:
        return _make_id(
            "write", thread_id, checkpoint_ns, checkpoint_id, task_id, str(idx)
        )

    # ------------------------------------------------------------------
    # Binary encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_blob(blob: bytes | None) -> str | None:
        if blob is None:
            return None
        return base64.b64encode(blob).decode("ascii")

    @staticmethod
    def _decode_blob(s: str | None) -> bytes:
        if not s:
            return b""
        return base64.b64decode(s)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        blob_values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[PointStruct]:
        """Build `PointStruct` list for the blob collection."""
        points: list[PointStruct] = []
        for channel, ver in versions.items():
            if channel in blob_values:
                btype, raw = self.serde.dumps_typed(blob_values[channel])
                encoded: str | None = self._encode_blob(raw)
            else:
                btype, encoded = "empty", None
            points.append(
                PointStruct(
                    id=self._blob_pid(
                        thread_id, checkpoint_ns, channel, cast(str, ver)
                    ),
                    vector=_DUMMY_VEC,
                    payload={
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "channel": channel,
                        "version": cast(str, ver),
                        "type": btype,
                        "blob": encoded,
                    },
                )
            )
        return points

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[PointStruct]:
        """Build `PointStruct` list for the writes collection."""
        points: list[PointStruct] = []
        for raw_idx, (channel, value) in enumerate(writes):
            idx = WRITES_IDX_MAP.get(channel, raw_idx)
            wtype, raw = self.serde.dumps_typed(value)
            points.append(
                PointStruct(
                    id=self._write_pid(
                        thread_id, checkpoint_ns, checkpoint_id, task_id, idx
                    ),
                    vector=_DUMMY_VEC,
                    payload={
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                        "task_path": task_path,
                        "idx": idx,
                        "channel": channel,
                        "type": wtype,
                        "blob": self._encode_blob(raw),
                    },
                )
            )
        return points

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_checkpoint_tuple(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        parent_checkpoint_id: str | None,
        checkpoint_json: str,
        metadata_json: str,
    ) -> CheckpointTuple:
        """Assemble a `CheckpointTuple` from raw Qdrant payload strings."""
        checkpoint_dict: dict[str, Any] = orjson.loads(checkpoint_json)
        channel_versions: dict[str, str] = checkpoint_dict.get("channel_versions", {})

        # 1. Batch-fetch blobs by deterministic ID.
        loaded_channels: dict[str, Any] = {}
        if channel_versions:
            blob_ids = [
                self._blob_pid(thread_id, checkpoint_ns, ch, ver)
                for ch, ver in channel_versions.items()
            ]
            blob_points = self._client.retrieve(
                collection_name=self._blob_collection,
                ids=blob_ids,
                with_payload=True,
                with_vectors=False,
            )
            blob_map = {str(p.id): p.payload for p in blob_points}
            for ch, ver in channel_versions.items():
                pid = self._blob_pid(thread_id, checkpoint_ns, ch, ver)
                payload = blob_map.get(pid)
                if payload is None:
                    # Value was stored inline in checkpoint_json (primitive).
                    continue
                btype = payload.get("type", "empty")
                if btype == "empty":
                    continue
                raw = self._decode_blob(payload.get("blob"))
                loaded_channels[ch] = self.serde.loads_typed((btype, raw))

        full_checkpoint: dict[str, Any] = {
            **checkpoint_dict,
            "channel_values": {
                **checkpoint_dict.get("channel_values", {}),
                **loaded_channels,
            },
        }

        # 2. Fetch all pending writes for this checkpoint.
        write_points: list[Any] = []
        offset: Any = None
        while True:
            pts, next_offset = self._client.scroll(
                collection_name=self._writes_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="thread_id", match=MatchValue(value=thread_id)
                        ),
                        FieldCondition(
                            key="checkpoint_ns",
                            match=MatchValue(value=checkpoint_ns),
                        ),
                        FieldCondition(
                            key="checkpoint_id",
                            match=MatchValue(value=checkpoint_id),
                        ),
                    ]
                ),
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            write_points.extend(pts)
            if next_offset is None:
                break
            offset = next_offset

        write_points.sort(key=lambda p: (p.payload["task_id"], p.payload["idx"]))
        pending_writes = [
            (
                p.payload["task_id"],
                p.payload["channel"],
                self.serde.loads_typed(
                    (p.payload["type"], self._decode_blob(p.payload.get("blob")))
                ),
            )
            for p in write_points
        ]

        # 3. Deserialise metadata.
        metadata = self.serde.loads_typed(("json", metadata_json.encode("utf-8")))

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=cast(Checkpoint, full_checkpoint),
            metadata=metadata,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
            pending_writes=pending_writes,
        )

    # ------------------------------------------------------------------
    # Synchronous CRUD (used directly by QdrantSaver and via executor
    # by AsyncQdrantSaver)
    # ------------------------------------------------------------------

    def _get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        self.setup()
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")

        if checkpoint_id:
            pts = self._client.retrieve(
                collection_name=self._chk_collection,
                ids=[self._checkpoint_pid(thread_id, checkpoint_ns, checkpoint_id)],
                with_payload=True,
                with_vectors=False,
            )
            if not pts:
                return None
            pl = pts[0].payload
        else:
            all_pts, _ = self._client.scroll(
                collection_name=self._chk_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="thread_id", match=MatchValue(value=thread_id)
                        ),
                        FieldCondition(
                            key="checkpoint_ns",
                            match=MatchValue(value=checkpoint_ns),
                        ),
                    ]
                ),
                limit=10_000,
                with_payload=True,
                with_vectors=False,
            )
            if not all_pts:
                return None
            # UUID6 checkpoint IDs are lexicographically time-ordered.
            all_pts.sort(
                key=lambda p: p.payload["checkpoint_id"] if p.payload else "",
                reverse=True,
            )
            pl = all_pts[0].payload

        assert pl is not None
        return self._load_checkpoint_tuple(
            thread_id=pl["thread_id"],
            checkpoint_ns=pl["checkpoint_ns"],
            checkpoint_id=pl["checkpoint_id"],
            parent_checkpoint_id=pl.get("parent_checkpoint_id"),
            checkpoint_json=pl["checkpoint_json"],
            metadata_json=pl["metadata_json"],
        )

    def _list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> list[CheckpointTuple]:
        self.setup()
        must_conditions: list[FieldCondition] = []

        if config:
            thread_id: str = config["configurable"]["thread_id"]
            must_conditions.append(
                FieldCondition(key="thread_id", match=MatchValue(value=thread_id))
            )
            checkpoint_ns = config["configurable"].get("checkpoint_ns")
            if checkpoint_ns is not None:
                must_conditions.append(
                    FieldCondition(
                        key="checkpoint_ns",
                        match=MatchValue(value=checkpoint_ns),
                    )
                )
            if cid := get_checkpoint_id(config):
                must_conditions.append(
                    FieldCondition(key="checkpoint_id", match=MatchValue(value=cid))
                )

        qdrant_filter = (
            Filter(must=cast(list[Any], must_conditions)) if must_conditions else None
        )

        all_pts: list[Any] = []
        offset: Any = None
        while True:
            pts, next_offset = self._client.scroll(
                collection_name=self._chk_collection,
                scroll_filter=qdrant_filter,
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_pts.extend(pts)
            if next_offset is None:
                break
            offset = next_offset

        # Sort newest first (UUID6 lexicographic order).
        all_pts.sort(
            key=lambda p: p.payload["checkpoint_id"] if p.payload else "",
            reverse=True,
        )

        # Metadata filter (containment — same semantics as postgres `@>`).
        if filter:

            def _matches_filter(pl: dict[str, Any]) -> bool:
                try:
                    meta: dict[str, Any] = orjson.loads(pl["metadata_json"])
                except Exception:
                    return False
                return all(meta.get(k) == v for k, v in filter.items())

            all_pts = [p for p in all_pts if _matches_filter(p.payload)]

        # `before` predicate — same as postgres `checkpoint_id < %s`.
        if before is not None:
            before_id = get_checkpoint_id(before)
            if before_id:
                all_pts = [p for p in all_pts if p.payload["checkpoint_id"] < before_id]

        if limit is not None:
            all_pts = all_pts[:limit]

        return [
            self._load_checkpoint_tuple(
                thread_id=p.payload["thread_id"],
                checkpoint_ns=p.payload["checkpoint_ns"],
                checkpoint_id=p.payload["checkpoint_id"],
                parent_checkpoint_id=p.payload.get("parent_checkpoint_id"),
                checkpoint_json=p.payload["checkpoint_json"],
                metadata_json=p.payload["metadata_json"],
            )
            for p in all_pts
        ]

    def _put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        self.setup()
        configurable = config["configurable"].copy()
        thread_id: str = configurable.pop("thread_id")
        checkpoint_ns: str = configurable.pop("checkpoint_ns", "")
        parent_checkpoint_id: str | None = configurable.pop("checkpoint_id", None)

        copy = checkpoint.copy()
        copy["channel_values"] = copy["channel_values"].copy()

        # Separate non-primitive values → stored in the blob collection.
        blob_values: dict[str, Any] = {}
        for k, v in list(checkpoint["channel_values"].items()):
            if v is not None and not isinstance(v, (str, int, float, bool)):
                blob_values[k] = copy["channel_values"].pop(k)

        # Upsert blobs.
        if blob_versions := {k: v for k, v in new_versions.items() if k in blob_values}:
            blob_points = self._dump_blobs(
                thread_id, checkpoint_ns, blob_values, blob_versions
            )
            self._client.upsert(
                collection_name=self._blob_collection, points=blob_points
            )

        # Serialise checkpoint and metadata as JSON strings.
        serialisable_meta = get_serializable_checkpoint_metadata(config, metadata)
        checkpoint_json = orjson.dumps(copy).decode("utf-8")
        metadata_json = orjson.dumps(serialisable_meta).decode("utf-8")

        self._client.upsert(
            collection_name=self._chk_collection,
            points=[
                PointStruct(
                    id=self._checkpoint_pid(thread_id, checkpoint_ns, checkpoint["id"]),
                    vector=_DUMMY_VEC,
                    payload={
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint["id"],
                        "parent_checkpoint_id": parent_checkpoint_id,
                        "checkpoint_json": checkpoint_json,
                        "metadata_json": metadata_json,
                    },
                )
            ],
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def _put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.setup()
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"]["checkpoint_ns"]
        checkpoint_id: str = config["configurable"]["checkpoint_id"]

        points = self._dump_writes(
            thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, writes
        )
        if not points:
            return

        # Mirror postgres ON CONFLICT logic:
        #   UPSERT_CHECKPOINT_WRITES_SQL (DO UPDATE) is used when all channels
        #   are in WRITES_IDX_MAP.  Otherwise INSERT_CHECKPOINT_WRITES_SQL
        #   (DO NOTHING) is used.
        if all(w[0] in WRITES_IDX_MAP for w in writes):
            # System channels — always overwrite.
            self._client.upsert(collection_name=self._writes_collection, points=points)
        else:
            # Mixed/task channels — only insert if not already present.
            existing = self._client.retrieve(
                collection_name=self._writes_collection,
                ids=[str(p.id) for p in points],
                with_payload=False,
                with_vectors=False,
            )
            existing_ids = {str(p.id) for p in existing}
            to_upsert: list[PointStruct] = []
            for pt, (channel, _) in zip(points, writes, strict=False):
                if channel in WRITES_IDX_MAP or str(pt.id) not in existing_ids:
                    to_upsert.append(pt)
            if to_upsert:
                self._client.upsert(
                    collection_name=self._writes_collection, points=to_upsert
                )

    def _delete_thread(self, thread_id: str) -> None:
        self.setup()
        filt = Filter(
            must=[FieldCondition(key="thread_id", match=MatchValue(value=thread_id))]
        )
        for coll in (
            self._chk_collection,
            self._blob_collection,
            self._writes_collection,
        ):
            self._client.delete(collection_name=coll, points_selector=filt)

    # ------------------------------------------------------------------
    # Version helper (identical to postgres)
    # ------------------------------------------------------------------

    def get_next_version(self, current: str | None, channel: Any) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

"""Async Qdrant checkpoint saver using `AsyncQdrantClient`."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_serializable_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from qdrant_client import AsyncQdrantClient, QdrantClient

from langgraph.checkpoint.qdrant.base import (
    _DUMMY_VEC,
    BaseQdrantSaver,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
)


class AsyncQdrantSaver(BaseQdrantSaver):
    """Async checkpoint saver backed by Qdrant using `AsyncQdrantClient`.

    All I/O operations are truly asynchronous; no thread-executor wrapping is
    needed.  Use this class in async-native code paths for best performance.

    !!! example "Basic usage"

        ```python
        from qdrant_client import AsyncQdrantClient
        from langgraph.checkpoint.qdrant.aio import AsyncQdrantSaver

        async with AsyncQdrantSaver.from_client(
            AsyncQdrantClient(url="http://localhost:6333")
        ) as checkpointer:
            await checkpointer.setup()
            graph = builder.compile(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": "t1"}}
            result = await graph.ainvoke(inputs, config)
        ```

    Args:
        async_client: An existing `AsyncQdrantClient` instance.
        serde: Optional custom serializer.
        prefix: Optional collection-name prefix.
    """

    _async_client: AsyncQdrantClient

    def __init__(
        self,
        async_client: AsyncQdrantClient,
        *,
        serde: SerializerProtocol | None = None,
        prefix: str = "",
    ) -> None:
        # Pass a dummy sync client to the base; async operations use _async_client.
        super().__init__(
            QdrantClient(":memory:"),
            serde=serde,
            prefix=prefix,
        )
        self._async_client = async_client

    @classmethod
    @asynccontextmanager
    async def from_client(
        cls,
        async_client: AsyncQdrantClient,
        *,
        serde: SerializerProtocol | None = None,
        prefix: str = "",
    ) -> AsyncIterator[AsyncQdrantSaver]:
        """Async context-manager factory.

        Args:
            async_client: An `AsyncQdrantClient` instance.
            serde: Optional custom serializer.
            prefix: Optional collection-name prefix.

        Yields:
            AsyncQdrantSaver: A configured async saver instance.
        """
        saver = cls(async_client, serde=serde, prefix=prefix)
        try:
            yield saver
        finally:
            await async_client.close()

    # ------------------------------------------------------------------
    # Async collection setup
    # ------------------------------------------------------------------

    async def setup(self) -> None:  # type: ignore[override]
        """Create collections and payload indexes asynchronously (idempotent)."""
        if self._is_setup:
            return
        existing_resp = await self._async_client.get_collections()
        existing = {c.name for c in existing_resp.collections}

        from qdrant_client.models import Distance, VectorParams

        dummy_vec_cfg = {"v": VectorParams(size=1, distance=Distance.DOT)}

        specs: list[tuple[str, list[str]]] = [
            (self._chk_collection, ["thread_id", "checkpoint_ns", "checkpoint_id"]),
            (self._blob_collection, ["thread_id", "checkpoint_ns"]),
            (
                self._writes_collection,
                ["thread_id", "checkpoint_ns", "checkpoint_id"],
            ),
        ]
        for name, payload_fields in specs:
            if name not in existing:
                await self._async_client.create_collection(
                    collection_name=name,
                    vectors_config=dummy_vec_cfg,
                )
            for field in payload_fields:
                await self._async_client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
        self._is_setup = True

    # ------------------------------------------------------------------
    # Async CRUD helpers
    # ------------------------------------------------------------------

    async def _aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        await self.setup()
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")

        if checkpoint_id:
            pts = await self._async_client.retrieve(
                collection_name=self._chk_collection,
                ids=[self._checkpoint_pid(thread_id, checkpoint_ns, checkpoint_id)],
                with_payload=True,
                with_vectors=False,
            )
            if not pts:
                return None
            pl = pts[0].payload
        else:
            all_pts, _ = await self._async_client.scroll(
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
            all_pts.sort(
                key=lambda p: p.payload["checkpoint_id"] if p.payload else "",
                reverse=True,
            )
            pl = all_pts[0].payload

        assert pl is not None
        return await self._aload_checkpoint_tuple(
            thread_id=pl["thread_id"],
            checkpoint_ns=pl["checkpoint_ns"],
            checkpoint_id=pl["checkpoint_id"],
            parent_checkpoint_id=pl.get("parent_checkpoint_id"),
            checkpoint_json=pl["checkpoint_json"],
            metadata_json=pl["metadata_json"],
        )

    async def _aload_checkpoint_tuple(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        parent_checkpoint_id: str | None,
        checkpoint_json: str,
        metadata_json: str,
    ) -> CheckpointTuple:
        import orjson

        checkpoint_dict: dict[str, Any] = orjson.loads(checkpoint_json)
        channel_versions: dict[str, str] = checkpoint_dict.get("channel_versions", {})

        loaded_channels: dict[str, Any] = {}
        if channel_versions:
            blob_ids = [
                self._blob_pid(thread_id, checkpoint_ns, ch, ver)
                for ch, ver in channel_versions.items()
            ]
            blob_points = await self._async_client.retrieve(
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

        write_points: list[Any] = []
        offset: Any = None
        while True:
            pts, next_offset = await self._async_client.scroll(
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
                    (
                        p.payload["type"],
                        self._decode_blob(p.payload.get("blob")),
                    )
                ),
            )
            for p in write_points
        ]

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
    # Async interface
    # ------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await self._aget_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        await self.setup()
        import orjson

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
            pts, next_offset = await self._async_client.scroll(
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

        all_pts.sort(key=lambda p: p.payload["checkpoint_id"], reverse=True)

        if filter:

            def _matches(pl: dict[str, Any]) -> bool:
                try:
                    meta: dict[str, Any] = orjson.loads(pl["metadata_json"])
                except Exception:
                    return False
                return all(meta.get(k) == v for k, v in filter.items())

            all_pts = [p for p in all_pts if _matches(p.payload)]

        if before is not None:
            before_id = get_checkpoint_id(before)
            if before_id:
                all_pts = [p for p in all_pts if p.payload["checkpoint_id"] < before_id]

        if limit is not None:
            all_pts = all_pts[:limit]

        for p in all_pts:
            yield await self._aload_checkpoint_tuple(
                thread_id=p.payload["thread_id"],
                checkpoint_ns=p.payload["checkpoint_ns"],
                checkpoint_id=p.payload["checkpoint_id"],
                parent_checkpoint_id=p.payload.get("parent_checkpoint_id"),
                checkpoint_json=p.payload["checkpoint_json"],
                metadata_json=p.payload["metadata_json"],
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        import orjson

        await self.setup()
        configurable = config["configurable"].copy()
        thread_id: str = configurable.pop("thread_id")
        checkpoint_ns: str = configurable.pop("checkpoint_ns", "")
        parent_checkpoint_id: str | None = configurable.pop("checkpoint_id", None)

        copy = checkpoint.copy()
        copy["channel_values"] = copy["channel_values"].copy()

        blob_values: dict[str, Any] = {}
        for k, v in list(checkpoint["channel_values"].items()):
            if v is not None and not isinstance(v, (str, int, float, bool)):
                blob_values[k] = copy["channel_values"].pop(k)

        if blob_versions := {k: v for k, v in new_versions.items() if k in blob_values}:
            blob_points = self._dump_blobs(
                thread_id, checkpoint_ns, blob_values, blob_versions
            )
            await self._async_client.upsert(
                collection_name=self._blob_collection, points=blob_points
            )

        serialisable_meta = get_serializable_checkpoint_metadata(config, metadata)
        checkpoint_json = orjson.dumps(copy).decode("utf-8")
        metadata_json = orjson.dumps(serialisable_meta).decode("utf-8")

        await self._async_client.upsert(
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

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        from langgraph.checkpoint.base import WRITES_IDX_MAP

        await self.setup()
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"]["checkpoint_ns"]
        checkpoint_id: str = config["configurable"]["checkpoint_id"]

        points = self._dump_writes(
            thread_id, checkpoint_ns, checkpoint_id, task_id, task_path, writes
        )
        if not points:
            return

        if all(w[0] in WRITES_IDX_MAP for w in writes):
            await self._async_client.upsert(
                collection_name=self._writes_collection, points=points
            )
        else:
            existing = await self._async_client.retrieve(
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
                await self._async_client.upsert(
                    collection_name=self._writes_collection, points=to_upsert
                )

    async def adelete_thread(self, thread_id: str) -> None:
        await self.setup()
        filt = Filter(
            must=[FieldCondition(key="thread_id", match=MatchValue(value=thread_id))]
        )
        for coll in (
            self._chk_collection,
            self._blob_collection,
            self._writes_collection,
        ):
            await self._async_client.delete(collection_name=coll, points_selector=filt)

    # ------------------------------------------------------------------
    # Sync fallbacks (delegate to thread executor)
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self._get_tuple(config)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        yield from self._list(config, filter=filter, before=before, limit=limit)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self._put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        self._delete_thread(thread_id)

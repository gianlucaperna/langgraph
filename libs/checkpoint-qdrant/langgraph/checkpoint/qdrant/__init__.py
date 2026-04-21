"""Synchronous Qdrant checkpoint saver."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from qdrant_client import QdrantClient

from langgraph.checkpoint.qdrant.base import BaseQdrantSaver


class QdrantSaver(BaseQdrantSaver):
    """Checkpoint saver backed by Qdrant.

    Stores checkpoints, channel blobs, and pending writes in three dedicated
    Qdrant collections (payload-only, no meaningful vectors).

    !!! example "Basic usage"

        ```python
        from qdrant_client import QdrantClient
        from langgraph.checkpoint.qdrant import QdrantSaver

        client = QdrantClient(url="http://localhost:6333")
        checkpointer = QdrantSaver(client)
        checkpointer.setup()  # idempotent — run once

        graph = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "thread-1"}}
        result = graph.invoke(inputs, config)
        ```

    !!! example "In-memory Qdrant (for testing)"

        ```python
        from qdrant_client import QdrantClient
        from langgraph.checkpoint.qdrant import QdrantSaver

        client = QdrantClient(":memory:")
        with QdrantSaver.from_client(client) as checkpointer:
            checkpointer.setup()
            ...
        ```

    Args:
        client: An existing `QdrantClient` instance.
        serde: Optional custom serializer. Defaults to `JsonPlusSerializer`.
        prefix: Optional string prepended to every collection name.
            Useful for multi-tenant scenarios.
    """

    def __init__(
        self,
        client: QdrantClient,
        *,
        serde: SerializerProtocol | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(client, serde=serde, prefix=prefix)

    @classmethod
    @contextmanager
    def from_client(
        cls,
        client: QdrantClient,
        *,
        serde: SerializerProtocol | None = None,
        prefix: str = "",
    ) -> Iterator[QdrantSaver]:
        """Context-manager factory that owns the client lifecycle.

        Args:
            client: A `QdrantClient` instance.
            serde: Optional custom serializer.
            prefix: Optional collection-name prefix.

        Yields:
            QdrantSaver: A configured saver instance.
        """
        saver = cls(client, serde=serde, prefix=prefix)
        try:
            yield saver
        finally:
            client.close()

    # ------------------------------------------------------------------
    # Sync interface
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Retrieve the checkpoint tuple that matches `config`.

        Args:
            config: Runnable configuration.  If `checkpoint_id` is absent the
                most recent checkpoint for the given thread is returned.

        Returns:
            The matching `CheckpointTuple`, or `None` if not found.
        """
        return self._get_tuple(config)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints ordered newest-first.

        Args:
            config: Filter by thread_id / checkpoint_ns.
            filter: Additional metadata containment filter.
            before: Return only checkpoints with an ID strictly less than this.
            limit: Maximum number of results.

        Yields:
            Matching `CheckpointTuple` objects.
        """
        yield from self._list(config, filter=filter, before=before, limit=limit)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist a checkpoint and return the updated config.

        Args:
            config: The config to associate with the checkpoint.
            checkpoint: The checkpoint to store.
            metadata: Metadata to store alongside the checkpoint.
            new_versions: Channel versions as of this write.

        Returns:
            Updated `RunnableConfig` containing the new `checkpoint_id`.
        """
        return self._put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate task writes linked to a checkpoint.

        Args:
            config: Configuration of the associated checkpoint.
            writes: List of `(channel, value)` pairs to store.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        self._put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes for the given thread.

        Args:
            thread_id: The thread whose data should be deleted.
        """
        self._delete_thread(thread_id)

    # ------------------------------------------------------------------
    # Async interface (delegated to thread executor)
    # ------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.get_running_loop().run_in_executor(
            None, self._get_tuple, config
        )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        tuples = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self._list(config, filter=filter, before=before, limit=limit),
        )
        for t in tuples:
            yield t

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.get_running_loop().run_in_executor(
            None, self._put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.get_running_loop().run_in_executor(
            None, self._put_writes, config, writes, task_id, task_path
        )

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.get_running_loop().run_in_executor(
            None, self._delete_thread, thread_id
        )

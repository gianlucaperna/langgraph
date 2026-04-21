"""Async Qdrant store backed by `AsyncBatchedBaseStore`.

All I/O is dispatched via a thread-pool executor around the synchronous
`QdrantClient`.  This keeps the implementation simple and correct while
providing non-blocking async convenience methods.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager

from langgraph.store.base import Op, Result, TTLConfig
from langgraph.store.base.batch import AsyncBatchedBaseStore
from qdrant_client import QdrantClient

from langgraph.store.qdrant.base import (
    BaseQdrantStore,
    QdrantIndexConfig,
    _ensure_index_config,
)

logger = logging.getLogger(__name__)


class AsyncQdrantStore(AsyncBatchedBaseStore, BaseQdrantStore):
    """Async Qdrant-backed store.

    Extends `AsyncBatchedBaseStore` so all async convenience methods
    (`aget`, `aput`, `asearch`, `alist_namespaces`, etc.) are available.
    All Qdrant I/O runs in a thread-pool executor around the synchronous
    `QdrantClient`, which is safe for concurrent use.

    !!! example "Basic usage"

        ```python
        from qdrant_client import QdrantClient
        from langgraph.store.qdrant import AsyncQdrantStore

        async with AsyncQdrantStore.from_client(
            QdrantClient(url="http://localhost:6333")
        ) as store:
            await store.setup()
            await store.aput(("users", "alice"), "prefs", {"theme": "dark"})
            item = await store.aget(("users", "alice"), "prefs")
        ```

    Args:
        client: An existing (synchronous) `QdrantClient` instance.
        collection_name: Qdrant collection name.
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
        AsyncBatchedBaseStore.__init__(self)
        self._client = client
        self._collection = collection_name
        self._ttl_config = ttl
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._ttl_sweeper_task: asyncio.Task[None] | None = None
        self._ttl_stop_event = asyncio.Event()

        if index is not None:
            self._embeddings, self._index_config = _ensure_index_config(index)
        else:
            self._embeddings = None
            self._index_config = None

    @classmethod
    @asynccontextmanager
    async def from_client(
        cls,
        client: QdrantClient,
        *,
        collection_name: str = "langgraph_store",
        index: QdrantIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> AsyncIterator[AsyncQdrantStore]:
        """Async context-manager factory.

        Args:
            client: A synchronous `QdrantClient` instance.
            collection_name: Qdrant collection name.
            index: Optional embedding/index configuration.
            ttl: Optional TTL configuration.

        Yields:
            AsyncQdrantStore: A configured async store instance.
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

    # ------------------------------------------------------------------
    # Async collection setup (runs sync setup in executor)
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Create the Qdrant collection and payload indexes (idempotent)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._setup_sync)

    # ------------------------------------------------------------------
    # abatch — called by AsyncBatchedBaseStore background task
    # ------------------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Sync batch execution — calls the direct sync implementation.

        Overrides `AsyncBatchedBaseStore.batch` (which would deadlock by
        routing back through `abatch`) with the plain `BaseQdrantStore.batch`.
        """
        return BaseQdrantStore.batch(self, ops)

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Process a batch of operations via thread executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, BaseQdrantStore.batch, self, list(ops)
        )

    # ------------------------------------------------------------------
    # Async TTL sweep
    # ------------------------------------------------------------------

    async def sweep_ttl_async(self) -> int:
        """Async wrapper around `sweep_ttl`."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.sweep_ttl)

    async def start_ttl_sweeper(self) -> None:
        """Start an asyncio task that periodically deletes expired items."""
        if self._ttl_config is None:
            return
        interval = self._ttl_config.get("sweep_interval_minutes")
        if interval is None:
            return
        self._ttl_stop_event.clear()
        self._ttl_sweeper_task = asyncio.create_task(
            self._ttl_sweep_loop_async(float(interval) * 60)
        )

    async def stop_ttl_sweeper(self) -> None:
        """Cancel the TTL sweeper task and wait for it to finish."""
        self._ttl_stop_event.set()
        if self._ttl_sweeper_task is not None:
            self._ttl_sweeper_task.cancel()
            try:
                await self._ttl_sweeper_task
            except asyncio.CancelledError:
                pass
            self._ttl_sweeper_task = None

    async def _ttl_sweep_loop_async(self, interval_secs: float) -> None:
        while not self._ttl_stop_event.is_set():
            try:
                await asyncio.sleep(interval_secs)
                n = await self.sweep_ttl_async()
                if n:
                    logger.debug("AsyncQdrantStore TTL sweep: deleted %d items", n)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("AsyncQdrantStore TTL sweep error")

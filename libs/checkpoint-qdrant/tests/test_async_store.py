# type: ignore
"""Async tests for AsyncQdrantStore."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import pytest
from qdrant_client import QdrantClient

from langgraph.store.qdrant import AsyncQdrantStore
from tests.embed_test_utils import CharacterEmbeddings

DIMS = 50
TTL_SECONDS = 6
TTL_MINUTES = TTL_SECONDS / 60


@asynccontextmanager
async def _async_store(
    index: dict | None = None,
    ttl: dict | None = None,
    collection_name: str = "test_async_store",
):
    # AsyncQdrantStore wraps a synchronous QdrantClient in an executor.
    client = QdrantClient(":memory:")
    async with AsyncQdrantStore.from_client(
        client,
        collection_name=collection_name,
        index=index,
        ttl=ttl,
    ) as store:
        await store.setup()
        yield store


@pytest.mark.asyncio
async def test_async_put_and_get() -> None:
    async with _async_store() as store:
        await store.aput(("ns", "a"), "k1", {"foo": "bar"})
        item = await store.aget(("ns", "a"), "k1")
        assert item is not None
        assert item.value == {"foo": "bar"}
        assert item.key == "k1"


@pytest.mark.asyncio
async def test_async_get_missing() -> None:
    async with _async_store() as store:
        assert await store.aget(("ns",), "missing") is None


@pytest.mark.asyncio
async def test_async_delete() -> None:
    async with _async_store() as store:
        await store.aput(("ns",), "k", {"x": 1})
        assert await store.aget(("ns",), "k") is not None
        await store.adelete(("ns",), "k")
        assert await store.aget(("ns",), "k") is None


@pytest.mark.asyncio
async def test_async_search_no_query() -> None:
    async with _async_store() as store:
        await store.aput(("docs",), "d1", {"text": "hello"})
        await store.aput(("docs",), "d2", {"text": "world"})

        results = await store.asearch(("docs",))
        keys = {r.key for r in results}
        assert "d1" in keys and "d2" in keys


@pytest.mark.asyncio
async def test_async_search_with_vector(
    fake_embeddings: CharacterEmbeddings,
) -> None:
    async with _async_store(
        index={"dims": DIMS, "embed": fake_embeddings, "fields": ["$"]},
        collection_name="avec_test",
    ) as store:
        await store.aput(("docs",), "py", {"text": "python programming"})
        await store.aput(("docs",), "ts", {"text": "typescript frontend"})

        results = await store.asearch(("docs",), query="python")
        assert len(results) >= 1
        assert results[0].key == "py"
        assert results[0].score is not None


@pytest.mark.asyncio
async def test_async_list_namespaces() -> None:
    async with _async_store(collection_name="ans_test") as store:
        await store.aput(("a", "1"), "k", {"v": 1})
        await store.aput(("b", "2"), "k", {"v": 2})

        namespaces = await store.alist_namespaces()
        ns_set = set(namespaces)
        assert ("a", "1") in ns_set
        assert ("b", "2") in ns_set


@pytest.mark.asyncio
async def test_async_concurrent_puts() -> None:
    """Multiple concurrent puts to different keys should all succeed."""
    async with _async_store(collection_name="conc_test") as store:
        tasks = [store.aput(("concurrent",), f"k{i}", {"i": i}) for i in range(10)]
        await asyncio.gather(*tasks)

        items = await store.asearch(("concurrent",), limit=20)
        assert len(items) == 10


@pytest.mark.asyncio
async def test_async_ttl_expiry() -> None:
    ttl_cfg = {
        "default_ttl": TTL_MINUTES,
        "refresh_on_read": False,
        "sweep_interval_minutes": TTL_MINUTES / 2,
    }
    async with _async_store(ttl=ttl_cfg, collection_name="attl_test") as store:
        await store.aput(("ns",), "expires", {"v": 1})

        assert await store.aget(("ns",), "expires") is not None

        time.sleep(TTL_SECONDS + 1)
        await store.sweep_ttl_async()

        assert await store.aget(("ns",), "expires") is None

# type: ignore
"""Async tests for AsyncQdrantSaver (using the sync-backed QdrantSaver async methods)."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.base import (
    empty_checkpoint,
)
from qdrant_client import QdrantClient

from langgraph.checkpoint.qdrant import QdrantSaver


@pytest.fixture
def saver():
    """Async-capable saver backed by in-memory Qdrant."""
    client = QdrantClient(":memory:")
    s = QdrantSaver(client)
    s.setup()
    yield s
    client.close()


@pytest.mark.asyncio
async def test_aget_tuple_missing(saver: QdrantSaver) -> None:
    result = await saver.aget_tuple(
        {"configurable": {"thread_id": "missing", "checkpoint_ns": ""}}
    )
    assert result is None


@pytest.mark.asyncio
async def test_aput_and_aget_tuple(saver: QdrantSaver) -> None:
    cfg = {"configurable": {"thread_id": "at1", "checkpoint_ns": ""}}
    chk = empty_checkpoint()
    saved_cfg = await saver.aput(cfg, chk, {}, {})

    result = await saver.aget_tuple(saved_cfg)
    assert result is not None
    assert result.checkpoint["id"] == chk["id"]


@pytest.mark.asyncio
async def test_alist(saver: QdrantSaver) -> None:
    cfg = {"configurable": {"thread_id": "at2", "checkpoint_ns": ""}}
    chk1 = empty_checkpoint()
    cfg1 = await saver.aput(cfg, chk1, {"source": "input"}, {})
    chk2 = empty_checkpoint()
    await saver.aput(cfg1, chk2, {"source": "loop"}, {})

    results = [
        t
        async for t in saver.alist(
            {"configurable": {"thread_id": "at2", "checkpoint_ns": ""}}
        )
    ]
    assert len(results) == 2
    # newest first
    assert results[0].checkpoint["id"] == chk2["id"]


@pytest.mark.asyncio
async def test_aput_writes(saver: QdrantSaver) -> None:
    cfg = {"configurable": {"thread_id": "at3", "checkpoint_ns": ""}}
    chk = empty_checkpoint()
    cfg = await saver.aput(cfg, chk, {}, {})

    await saver.aput_writes(cfg, [("ch", "val")], task_id="task-1")

    result = await saver.aget_tuple(cfg)
    assert result is not None
    writes = result.pending_writes or []
    assert any(w[1] == "ch" for w in writes)


@pytest.mark.asyncio
async def test_adelete_thread(saver: QdrantSaver) -> None:
    cfg = {"configurable": {"thread_id": "at4", "checkpoint_ns": ""}}
    await saver.aput(cfg, empty_checkpoint(), {}, {})

    assert await saver.aget_tuple(cfg) is not None

    await saver.adelete_thread("at4")

    result = await saver.aget_tuple(
        {"configurable": {"thread_id": "at4", "checkpoint_ns": ""}}
    )
    assert result is None

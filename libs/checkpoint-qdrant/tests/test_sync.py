# type: ignore
"""Synchronous tests for QdrantSaver."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    create_checkpoint,
    empty_checkpoint,
)
from qdrant_client import QdrantClient

from langgraph.checkpoint.qdrant import QdrantSaver


@contextmanager
def _saver(prefix: str = ""):
    """Create a QdrantSaver backed by an in-memory Qdrant instance."""
    client = QdrantClient(":memory:")
    saver = QdrantSaver(client, prefix=prefix)
    saver.setup()
    try:
        yield saver
    finally:
        client.close()


@pytest.fixture
def test_data():
    config_1: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_id": "1",
            "checkpoint_ns": "",
        }
    }
    config_2: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2",
            "checkpoint_ns": "",
        }
    }
    config_3: RunnableConfig = {
        "configurable": {
            "thread_id": "thread-2",
            "checkpoint_id": "2-inner",
            "checkpoint_ns": "inner",
        }
    }

    chkpnt_1: Checkpoint = empty_checkpoint()
    chkpnt_2: Checkpoint = create_checkpoint(chkpnt_1, {}, 1)
    chkpnt_3: Checkpoint = empty_checkpoint()

    metadata_1: CheckpointMetadata = {"source": "input", "step": 2}
    metadata_2: CheckpointMetadata = {"source": "loop", "step": 1}
    metadata_3: CheckpointMetadata = {}

    return {
        "configs": [config_1, config_2, config_3],
        "checkpoints": [chkpnt_1, chkpnt_2, chkpnt_3],
        "metadata": [metadata_1, metadata_2, metadata_3],
    }


def test_setup_is_idempotent() -> None:
    client = QdrantClient(":memory:")
    saver = QdrantSaver(client)
    saver.setup()
    saver.setup()  # second call must not raise
    client.close()


def test_put_and_get_tuple(test_data) -> None:
    configs = test_data["configs"]
    checkpoints = test_data["checkpoints"]
    metadata = test_data["metadata"]

    with _saver() as saver:
        # put() returns the updated config whose checkpoint_id is the new checkpoint UUID.
        saved_cfg = saver.put(configs[0], checkpoints[0], metadata[0], {})
        result = saver.get_tuple(saved_cfg)
        assert result is not None
        assert result.config["configurable"]["checkpoint_id"] == checkpoints[0]["id"]
        assert result.checkpoint["id"] == checkpoints[0]["id"]


def test_get_tuple_returns_none_for_missing(test_data) -> None:
    with _saver() as saver:
        result = saver.get_tuple(
            {"configurable": {"thread_id": "nonexistent", "checkpoint_ns": ""}}
        )
        assert result is None


def test_get_tuple_latest(test_data) -> None:
    """Without checkpoint_id, the most recent checkpoint is returned."""
    with _saver() as saver:
        checkpoints = test_data["checkpoints"]
        metadata = test_data["metadata"]

        cfg1 = saver.put(
            {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}},
            checkpoints[0],
            metadata[0],
            {},
        )
        saver.put(
            cfg1,
            checkpoints[1],
            metadata[1],
            {},
        )

        latest = saver.get_tuple(
            {"configurable": {"thread_id": "t1", "checkpoint_ns": ""}}
        )
        assert latest is not None
        assert latest.checkpoint["id"] == checkpoints[1]["id"]


def test_list_checkpoints(test_data) -> None:
    checkpoints = test_data["checkpoints"]
    metadata = test_data["metadata"]

    with _saver() as saver:
        cfg1 = saver.put(
            {"configurable": {"thread_id": "t2", "checkpoint_ns": ""}},
            checkpoints[0],
            metadata[0],
            {},
        )
        saver.put(cfg1, checkpoints[1], metadata[1], {})

        results = list(
            saver.list({"configurable": {"thread_id": "t2", "checkpoint_ns": ""}})
        )
        assert len(results) == 2
        # Newest first
        assert results[0].checkpoint["id"] == checkpoints[1]["id"]
        assert results[1].checkpoint["id"] == checkpoints[0]["id"]


def test_list_with_limit(test_data) -> None:
    with _saver() as saver:
        cfg = {"configurable": {"thread_id": "t3", "checkpoint_ns": ""}}
        for _i in range(5):
            chk = empty_checkpoint()
            cfg = saver.put(cfg, chk, {}, {})

        results = list(
            saver.list(
                {"configurable": {"thread_id": "t3", "checkpoint_ns": ""}},
                limit=3,
            )
        )
        assert len(results) == 3


def test_list_with_metadata_filter(test_data) -> None:
    with _saver() as saver:
        cfg = {"configurable": {"thread_id": "t4", "checkpoint_ns": ""}}
        chk1 = empty_checkpoint()
        chk2 = empty_checkpoint()
        cfg = saver.put(cfg, chk1, {"source": "input"}, {})
        cfg = saver.put(cfg, chk2, {"source": "loop"}, {})

        results = list(
            saver.list(
                {"configurable": {"thread_id": "t4", "checkpoint_ns": ""}},
                filter={"source": "input"},
            )
        )
        assert len(results) == 1
        assert results[0].checkpoint["id"] == chk1["id"]


def test_put_writes_and_get_tuple(test_data) -> None:
    with _saver() as saver:
        cfg = {"configurable": {"thread_id": "t5", "checkpoint_ns": ""}}
        chk = empty_checkpoint()
        cfg = saver.put(cfg, chk, {}, {})

        saver.put_writes(cfg, [("channel1", "value1")], task_id="task-1")

        result = saver.get_tuple(cfg)
        assert result is not None
        writes = result.pending_writes or []
        channels = [w[1] for w in writes]
        assert "channel1" in channels


def test_delete_thread(test_data) -> None:
    with _saver() as saver:
        cfg = {"configurable": {"thread_id": "t6", "checkpoint_ns": ""}}
        chk = empty_checkpoint()
        saver.put(cfg, chk, {}, {})

        result = saver.get_tuple(cfg)
        assert result is not None

        saver.delete_thread("t6")

        result_after = saver.get_tuple(
            {"configurable": {"thread_id": "t6", "checkpoint_ns": ""}}
        )
        assert result_after is None


def test_multiple_namespaces(test_data) -> None:
    with _saver() as saver:
        saver.put(
            {"configurable": {"thread_id": "t7", "checkpoint_ns": ""}},
            empty_checkpoint(),
            {},
            {},
        )
        saver.put(
            {"configurable": {"thread_id": "t7", "checkpoint_ns": "inner"}},
            empty_checkpoint(),
            {},
            {},
        )

        result_root = saver.get_tuple(
            {"configurable": {"thread_id": "t7", "checkpoint_ns": ""}}
        )
        result_inner = saver.get_tuple(
            {"configurable": {"thread_id": "t7", "checkpoint_ns": "inner"}}
        )
        assert result_root is not None
        assert result_inner is not None
        assert result_root.checkpoint["id"] != result_inner.checkpoint["id"]


def test_prefix_isolation() -> None:
    """Two savers with different prefixes must not see each other's data."""
    client = QdrantClient(":memory:")
    saver_a = QdrantSaver(client, prefix="a_")
    saver_b = QdrantSaver(client, prefix="b_")
    saver_a.setup()
    saver_b.setup()

    cfg = {"configurable": {"thread_id": "shared", "checkpoint_ns": ""}}
    saver_a.put(cfg, empty_checkpoint(), {}, {})

    result_a = saver_a.get_tuple(cfg)
    result_b = saver_b.get_tuple(cfg)

    assert result_a is not None
    assert result_b is None
    client.close()


def test_list_before_filter() -> None:
    with _saver() as saver:
        cfg = {"configurable": {"thread_id": "t8", "checkpoint_ns": ""}}
        chk1 = empty_checkpoint()
        cfg1 = saver.put(cfg, chk1, {}, {})
        chk2 = empty_checkpoint()
        cfg2 = saver.put(cfg1, chk2, {}, {})
        chk3 = empty_checkpoint()
        cfg3 = saver.put(cfg2, chk3, {}, {})

        results = list(
            saver.list(
                {"configurable": {"thread_id": "t8", "checkpoint_ns": ""}},
                before=cfg3,
            )
        )
        # Should return chk2 and chk1 (those before cfg3)
        checkpoint_ids = [r.checkpoint["id"] for r in results]
        assert chk3["id"] not in checkpoint_ids
        assert chk2["id"] in checkpoint_ids
        assert chk1["id"] in checkpoint_ids

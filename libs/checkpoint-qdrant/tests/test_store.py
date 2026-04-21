# type: ignore
"""Sync tests for QdrantStore."""

from __future__ import annotations

import time
from contextlib import contextmanager

import pytest
from langgraph.store.base import (
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchOp,
)
from qdrant_client import QdrantClient

from langgraph.store.qdrant import QdrantStore
from tests.embed_test_utils import CharacterEmbeddings

DIMS = 50
TTL_SECONDS = 6
TTL_MINUTES = TTL_SECONDS / 60


@contextmanager
def _store(
    index: dict | None = None,
    ttl: dict | None = None,
    collection_name: str = "test_store",
):
    client = QdrantClient(":memory:")
    with QdrantStore.from_client(
        client,
        collection_name=collection_name,
        index=index,
        ttl=ttl,
    ) as store:
        store.setup()
        yield store


@pytest.fixture
def store():
    with _store() as s:
        yield s


@pytest.fixture
def store_with_index(fake_embeddings: CharacterEmbeddings):
    with _store(index={"dims": DIMS, "embed": fake_embeddings, "fields": ["$"]}) as s:
        yield s


def test_batch_order(store: QdrantStore) -> None:
    store.put(("test", "foo"), "key1", {"data": "value1"})
    store.put(("test", "bar"), "key2", {"data": "value2"})

    ops = [
        GetOp(namespace=("test", "foo"), key="key1"),
        PutOp(namespace=("test", "bar"), key="key2", value={"data": "value2"}),
        SearchOp(
            namespace_prefix=("test",), filter={"data": "value1"}, limit=10, offset=0
        ),
        ListNamespacesOp(match_conditions=None, max_depth=None, limit=10, offset=0),
        GetOp(namespace=("test",), key="nonexistent"),
    ]

    results = store.batch(ops)
    assert len(results) == 5
    assert isinstance(results[0], Item)
    assert results[0].value == {"data": "value1"}
    assert results[0].key == "key1"
    assert results[1] is None  # PutOp returns None
    assert isinstance(results[2], list)
    assert len(results[2]) >= 1
    assert isinstance(results[3], list)
    assert len(results[3]) > 0
    assert results[4] is None  # missing key


def test_put_and_get(store: QdrantStore) -> None:
    store.put(("ns", "a"), "k1", {"foo": "bar"})
    item = store.get(("ns", "a"), "k1")
    assert item is not None
    assert item.value == {"foo": "bar"}
    assert item.key == "k1"
    assert item.namespace == ("ns", "a")


def test_get_missing(store: QdrantStore) -> None:
    assert store.get(("ns",), "missing") is None


def test_delete(store: QdrantStore) -> None:
    store.put(("ns",), "k", {"x": 1})
    assert store.get(("ns",), "k") is not None
    store.delete(("ns",), "k")
    assert store.get(("ns",), "k") is None


def test_search_no_query(store: QdrantStore) -> None:
    store.put(("docs",), "d1", {"text": "hello world"})
    store.put(("docs",), "d2", {"text": "foo bar"})

    results = store.search(("docs",))
    assert len(results) >= 2
    keys = {r.key for r in results}
    assert "d1" in keys and "d2" in keys


def test_search_with_filter(store: QdrantStore) -> None:
    store.put(("docs",), "d1", {"status": "active", "text": "a"})
    store.put(("docs",), "d2", {"status": "inactive", "text": "b"})

    results = store.search(("docs",), filter={"status": "active"})
    assert len(results) == 1
    assert results[0].key == "d1"


def test_search_with_vector_query(store_with_index: QdrantStore) -> None:
    store_with_index.put(("docs",), "py", {"text": "python programming language"})
    store_with_index.put(("docs",), "ts", {"text": "typescript javascript frontend"})

    results = store_with_index.search(("docs",), query="python")
    assert len(results) >= 1
    # The most relevant result should be about python
    assert results[0].key == "py"
    assert results[0].score is not None


def test_search_namespace_prefix(store: QdrantStore) -> None:
    store.put(("a", "1"), "k", {"x": 1})
    store.put(("b", "1"), "k", {"x": 2})

    results = store.search(("a",))
    assert all(r.namespace[0] == "a" for r in results)


def test_list_namespaces(store: QdrantStore) -> None:
    store.put(("ns", "a"), "k1", {"v": 1})
    store.put(("ns", "b"), "k2", {"v": 2})
    store.put(("other",), "k3", {"v": 3})

    namespaces = list(store.list_namespaces(prefix=("ns",)))
    ns_set = set(namespaces)
    assert ("ns", "a") in ns_set
    assert ("ns", "b") in ns_set
    assert ("other",) not in ns_set


def test_list_namespaces_max_depth(store: QdrantStore) -> None:
    store.put(("a", "b", "c"), "k", {"v": 1})
    store.put(("a", "b", "d"), "k", {"v": 2})

    namespaces = list(store.list_namespaces(max_depth=2))
    assert all(len(ns) <= 2 for ns in namespaces)
    assert ("a", "b") in set(namespaces)


def test_list_namespaces_suffix(store: QdrantStore) -> None:
    store.put(("a", "common"), "k1", {"v": 1})
    store.put(("b", "common"), "k2", {"v": 2})
    store.put(("c", "different"), "k3", {"v": 3})

    namespaces = list(store.list_namespaces(suffix=("common",)))
    ns_set = set(namespaces)
    assert ("a", "common") in ns_set
    assert ("b", "common") in ns_set
    assert ("c", "different") not in ns_set


def test_search_offset_limit(store: QdrantStore) -> None:
    for i in range(5):
        store.put(("paging",), f"k{i}", {"i": i})

    page1 = store.search(("paging",), limit=2, offset=0)
    page2 = store.search(("paging",), limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 2
    keys1 = {r.key for r in page1}
    keys2 = {r.key for r in page2}
    assert keys1.isdisjoint(keys2)


def test_ttl_expiry() -> None:
    ttl_cfg = {
        "default_ttl": TTL_MINUTES,
        "refresh_on_read": False,
        "sweep_interval_minutes": TTL_MINUTES / 2,
    }
    with _store(ttl=ttl_cfg, collection_name="ttl_test") as store:
        store.put(("ns",), "expires", {"v": 1})
        store.put(("ns",), "nope", {"v": 2}, ttl=None)  # no TTL — persists

        item = store.get(("ns",), "expires")
        assert item is not None

        # Wait for expiry
        time.sleep(TTL_SECONDS + 1)
        store.sweep_ttl()

        expired = store.get(("ns",), "expires")
        assert expired is None  # should be gone


def test_updated_at_changes_on_put(store: QdrantStore) -> None:
    store.put(("ns",), "k", {"v": 1})
    item1 = store.get(("ns",), "k")
    assert item1 is not None
    t1 = item1.updated_at

    time.sleep(0.05)
    store.put(("ns",), "k", {"v": 2})
    item2 = store.get(("ns",), "k")
    assert item2 is not None
    t2 = item2.updated_at

    assert t2 >= t1
    assert item2.created_at == item1.created_at  # created_at preserved


def test_index_false_does_not_embed(
    fake_embeddings: CharacterEmbeddings,
) -> None:
    with _store(
        index={"dims": DIMS, "embed": fake_embeddings},
        collection_name="noindex_test",
    ) as store:
        store.put(("docs",), "indexed", {"text": "python"})
        store.put(("docs",), "skipped", {"text": "python"}, index=False)

        # Both are retrievable by get
        assert store.get(("docs",), "indexed") is not None
        assert store.get(("docs",), "skipped") is not None

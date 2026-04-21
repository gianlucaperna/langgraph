"""Shared test fixtures for checkpoint-qdrant tests."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from tests.embed_test_utils import CharacterEmbeddings

DEFAULT_QDRANT_URL = "http://localhost:6333"
DIMS = 50


@pytest.fixture(scope="function")
def qdrant_client() -> QdrantClient:
    """In-process Qdrant client backed by an in-memory instance."""
    return QdrantClient(":memory:")


@pytest.fixture(scope="function")
def remote_qdrant_client() -> QdrantClient:
    """Qdrant client pointing at the Docker-Compose Qdrant instance."""
    return QdrantClient(url=DEFAULT_QDRANT_URL)


@pytest.fixture
def fake_embeddings() -> CharacterEmbeddings:
    return CharacterEmbeddings(dims=DIMS)

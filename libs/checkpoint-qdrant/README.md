# LangGraph Checkpoint Qdrant

Qdrant-based implementations of LangGraph's checkpoint saver and long-term memory store.

## Installation

```bash
pip install langgraph-checkpoint-qdrant
```

## Checkpoint Saver

`QdrantSaver` / `AsyncQdrantSaver` store LangGraph checkpoints in Qdrant collections.

```python
from qdrant_client import QdrantClient
from langgraph.checkpoint.qdrant import QdrantSaver

client = QdrantClient(url="http://localhost:6333")
checkpointer = QdrantSaver(client)
checkpointer.setup()  # create collections + indexes (idempotent)

graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "thread-1"}}
graph.invoke(inputs, config)
```

## Long-term Memory Store

`QdrantStore` / `AsyncQdrantStore` implement `BaseStore` with optional vector similarity
search, namespace filtering, and TTL expiration.

```python
from qdrant_client import QdrantClient
from langgraph.store.qdrant import QdrantStore

client = QdrantClient(url="http://localhost:6333")

with QdrantStore.from_client(
    client,
    index={
        "dims": 1536,
        "embed": my_embedding_fn,
        "fields": ["text"],
    },
) as store:
    store.setup()

    store.put(("users", "alice"), "prefs", {"theme": "dark", "text": "prefers dark mode"})
    results = store.search(("users",), query="dark mode", limit=5)
```

### Async usage

```python
from qdrant_client import AsyncQdrantClient
from langgraph.store.qdrant import AsyncQdrantStore

async with AsyncQdrantStore.from_client(
    AsyncQdrantClient(url="http://localhost:6333"),
    index={"dims": 1536, "embed": my_async_embed_fn},
) as store:
    await store.setup()
    await store.aput(("docs",), "doc1", {"text": "hello world"})
    items = await store.asearch(("docs",), query="hello")
```

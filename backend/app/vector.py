"""Qdrant access (local on-disk mode by default, no Docker needed).

Collections
-----------
doc_chunks     uploaded-PDF chunks   {run_id, filename, page, heading, text}
entities       entity embeddings     {name, type, last_run}      (shared across runs)
report_claims  claims of past reports {report_id, claim, url, title, date, credibility, paragraph}
fc_<uuid>      TEMPORARY collection created by the fact-checker per source page

LEARNING NOTE: the temporary collection is how the fact-checker satisfies the
"cosine similarity via Qdrant" requirement: embed the page's sentences, upsert,
query with the claim vector, read the top score, delete the collection.
"""
import logging
import threading
import uuid
from contextlib import contextmanager

import numpy as np

from app.config import settings
from app.embeddings import DIM

log = logging.getLogger("insightforge.vector")
_client = None
_lock = threading.RLock()  # local-mode qdrant is not safe for concurrent use
_NS = uuid.UUID("12345678-1234-5678-1234-567812345678")


def client():
    global _client
    with _lock:
        if _client is None:
            from qdrant_client import QdrantClient

            if settings.qdrant_url:
                _client = QdrantClient(url=settings.qdrant_url)
            elif settings.qdrant_path == ":memory:":
                _client = QdrantClient(":memory:")
            else:
                _client = QdrantClient(path=settings.qdrant_path)
        return _client


def stable_id(*parts: str) -> str:
    """Deterministic UUID so re-indexing the same entity overwrites instead of duplicating."""
    return str(uuid.uuid5(_NS, "|".join(parts)))


def ensure(name: str) -> None:
    from qdrant_client.models import Distance, VectorParams

    with _lock:
        c = client()
        existing = {col.name for col in c.get_collections().collections}
        if name not in existing:
            c.create_collection(name, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))


def upsert(collection: str, vectors: np.ndarray, payloads: list[dict], ids: list[str] | None = None) -> list[str]:
    from qdrant_client.models import PointStruct

    ensure(collection)
    ids = ids or [str(uuid.uuid4()) for _ in payloads]
    points = [PointStruct(id=i, vector=v.tolist(), payload=p) for i, v, p in zip(ids, vectors, payloads)]
    with _lock:
        for k in range(0, len(points), 128):
            client().upsert(collection, points=points[k : k + 128])
    return ids


def search(collection: str, vector: np.ndarray, limit: int = 8, where: dict | None = None) -> list[dict]:
    """Return [{id, score, payload}] best-first. `where` = exact-match payload filter."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    ensure(collection)
    flt = None
    if where:
        flt = Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in where.items()])
    with _lock:
        res = client().query_points(
            collection, query=vector.tolist(), limit=limit, query_filter=flt, with_payload=True
        ).points
    return [{"id": str(p.id), "score": float(p.score), "payload": p.payload or {}} for p in res]


def get_payload(collection: str, point_id: str) -> dict | None:
    ensure(collection)
    with _lock:
        got = client().retrieve(collection, ids=[point_id], with_payload=True)
    return (got[0].payload or {}) if got else None


@contextmanager
def temp_collection(vectors: np.ndarray, payloads: list[dict]):
    """Create a throw-away collection, yield its name, always delete it afterwards."""
    name = f"fc_{uuid.uuid4().hex[:12]}"
    try:
        upsert(name, vectors, payloads)
        yield name
    finally:
        try:
            with _lock:
                client().delete_collection(name)
        except Exception:  # noqa: BLE001
            log.warning("could not delete temp collection %s", name)

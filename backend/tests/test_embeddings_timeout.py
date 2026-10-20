import sys
import time
import types

from app import embeddings
from app.config import settings


def test_hung_model_load_falls_back_to_hash_instead_of_blocking_forever():
    class SlowModel:
        def __init__(self, *a, **k):
            time.sleep(5)          # simulates a model download/check that never returns in time

    fake = types.ModuleType("fastembed")
    fake.TextEmbedding = SlowModel
    saved = (sys.modules.get("fastembed"), embeddings._backend, embeddings._model, embeddings._LOAD_TIMEOUT,
             embeddings._OFFLINE_TIMEOUT, settings.embeddings_backend)
    try:
        sys.modules["fastembed"] = fake
        embeddings._backend = embeddings._model = None
        embeddings._LOAD_TIMEOUT = embeddings._OFFLINE_TIMEOUT = 0.5
        settings.embeddings_backend = "fastembed"
        t0 = time.time()
        assert embeddings.backend_name() == "hash"
        assert time.time() - t0 < 3, "load must give up quickly, not hang"
        t1 = time.time()
        assert embeddings.embed(["hello"]).shape == (1, 384)   # later calls are instant
        assert time.time() - t1 < 1
    finally:
        old_mod, embeddings._backend, embeddings._model, embeddings._LOAD_TIMEOUT, embeddings._OFFLINE_TIMEOUT, settings.embeddings_backend = saved
        if old_mod is None:
            sys.modules.pop("fastembed", None)
        else:
            sys.modules["fastembed"] = old_mod

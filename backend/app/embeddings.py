"""Text embeddings. Default: fastembed (ONNX, no PyTorch). Fallback: hashed bag-of-words.

LEARNING NOTE: an embedding turns text into a vector so "similar meaning" becomes
"small angle between vectors". We L2-normalise every vector, so cosine similarity
is just a dot product. The same embedder is used for: document retrieval, claim
de-duplication, fact-checking and the diff agent.

The `hash` backend exists so tests/CI can run offline. It only captures word
overlap, so never use it for real runs (set EMBEDDINGS_BACKEND=fastembed).
"""
import hashlib
import logging
import os
import re
import threading

import numpy as np

from app.config import settings

log = logging.getLogger("insightforge.embed")
DIM = 384  # bge-small-en-v1.5 and the hash fallback both use 384 dims
_lock = threading.Lock()
_model = None
_backend = None


def _hash_embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        words = re.findall(r"[a-z0-9$%.]+", t.lower())
        grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for g in grams:
            h = int(hashlib.md5(g.encode()).hexdigest()[:8], 16)
            out[i, h % DIM] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


_LOAD_TIMEOUT = float(os.environ.get("EMBEDDING_LOAD_TIMEOUT", "75"))
_OFFLINE_TIMEOUT = float(os.environ.get("EMBEDDING_OFFLINE_TIMEOUT", "30"))


def _try_load(offline: bool, timeout: float):
    """Create the fastembed model in a helper thread so a hang can never block the pipeline.
    Returns (model_or_None, reason). On timeout the helper thread is abandoned (daemon)."""
    holder: dict = {}

    def work():
        try:
            from fastembed import TextEmbedding

            kwargs = {"local_files_only": True} if offline else {}
            holder["m"] = TextEmbedding(model_name=settings.embedding_model, **kwargs)
        except Exception as e:  # noqa: BLE001
            holder["err"] = e

    t = threading.Thread(target=work, daemon=True, name="embedding-load")
    t.start()
    t.join(timeout)
    if "m" in holder:
        return holder["m"], ""
    if "err" in holder:
        return None, f"{type(holder['err']).__name__}: {str(holder['err'])[:150]}"
    return None, f"timed out after {timeout:.0f}s"


def _load():
    """Load the embedding model ONCE, never blocking forever.
    1) try the locally cached model only (no network)   2) try a normal load (may download)
    3) fall back to the weak hash embedder and say so loudly."""
    global _model, _backend
    if _backend:
        return
    with _lock:
        if _backend:
            return
        if settings.embeddings_backend == "fastembed":
            for offline, timeout in ((True, _OFFLINE_TIMEOUT), (False, _LOAD_TIMEOUT)):
                model, reason = _try_load(offline, timeout)
                if model is not None:
                    _model, _backend = model, "fastembed"
                    log.info("embedding model loaded (%s)", "cached" if offline else "downloaded/online")
                    return
                log.warning("embedding load (%s) failed: %s", "offline" if offline else "online", reason)
            log.error("fastembed could not be loaded -> using HASH embeddings (low quality; fact-check and "
                      "dedupe are weaker). Fix the network/model cache and restart for real results.")
        _backend = "hash"


def embed(texts: list[str]) -> np.ndarray:
    """Return an (n, 384) array of L2-normalised vectors."""
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    _load()
    if _backend == "fastembed":
        with _lock:  # onnx session is not guaranteed thread-safe
            vecs = np.array(list(_model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms
    return _hash_embed(texts)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T


def backend_name() -> str:
    _load()
    return _backend or "unknown"

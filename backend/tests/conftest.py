"""Test setup: isolated temp data dir + offline hash embeddings (no model download, no network)."""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="insightforge_test_")
os.environ["DATA_DIR"] = _tmp
os.environ["QDRANT_PATH"] = ":memory:"
os.environ["EMBEDDINGS_BACKEND"] = "hash"
os.environ["GROQ_API_KEY"] = "test-key-not-used"

from app import db  # noqa: E402

db.init_db()

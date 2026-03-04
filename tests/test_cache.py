import os
import tempfile
import numpy as np
import pytest
from unittest.mock import patch
from embedder import _cache_key, load_cached, save_cache


@pytest.fixture
def temp_cache(tmp_path):
    with patch("embedder.CACHE_DIR", str(tmp_path)):
        yield tmp_path


@pytest.fixture
def dummy_pdf(tmp_path):
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"fake pdf content")
    return str(pdf)


@pytest.fixture
def dummy_data():
    chunks = ["chunk one", "chunk two", "chunk three"]
    embeddings = np.random.rand(3, 384).astype("float32")
    return chunks, embeddings


def test_save_and_load_cache(temp_cache, dummy_pdf, dummy_data):
    chunks, embeddings = dummy_data
    save_cache(dummy_pdf, chunks, embeddings)
    result = load_cached(dummy_pdf)
    assert result is not None
    loaded_chunks, loaded_embeddings = result
    assert loaded_chunks == chunks
    np.testing.assert_array_almost_equal(loaded_embeddings, embeddings)


def test_cache_miss_on_different_config(temp_cache, dummy_pdf, dummy_data):
    chunks, embeddings = dummy_data
    save_cache(dummy_pdf, chunks, embeddings)

    with patch("embedder.CHUNK_SIZE", 9999):
        result = load_cached(dummy_pdf)
    assert result is None


def test_cache_miss_no_file(temp_cache, dummy_pdf):
    result = load_cached(dummy_pdf)
    assert result is None

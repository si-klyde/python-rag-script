import numpy as np
import pytest
from embedder import (
    get_embeddings, build_index, search,
    mmr_search, bm25_search, merge_results, rerank,
)


@pytest.fixture
def sample_chunks():
    return [
        {"text": "Hello world", "metadata": {"page": 1}},
        {"text": "Machine learning is great", "metadata": {"page": 1}},
        {"text": "Python programming", "metadata": {"page": 2}},
    ]


@pytest.fixture
def sample_texts():
    return ["Hello world", "Machine learning is great", "Python programming"]


def test_get_embeddings_shape(sample_texts):
    embeddings = get_embeddings(sample_texts)
    assert embeddings.shape[0] == len(sample_texts)
    assert embeddings.shape[1] > 0
    assert embeddings.dtype == np.float32


def test_build_index(sample_texts):
    embeddings = get_embeddings(sample_texts)
    index = build_index(embeddings)
    assert index.ntotal == len(sample_texts)


def test_search_returns_tuples(sample_chunks):
    texts = [c["text"] for c in sample_chunks]
    embeddings = get_embeddings(texts)
    index = build_index(embeddings)
    query_emb = get_embeddings(["Hello"])
    results = search(
        index, query_emb[0], chunks=sample_chunks,
        chunk_embeddings=embeddings, query_text="Hello", top_k=2,
    )
    assert len(results) <= 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    assert all(isinstance(r[1], float) for r in results)
    indices = [r[0] for r in results]
    assert 0 in indices  # "Hello world" chunk should match "Hello"


# --- MMR tests ---


def test_mmr_search_diversity():
    """MMR should return more diverse results than raw top-K."""
    # Create embeddings: 3 very similar + 1 different
    similar = np.array([[1.0, 0.0, 0.0]] * 3, dtype="float32")
    similar[1] += np.array([0.01, 0.0, 0.0])  # near-duplicate
    similar[2] += np.array([0.02, 0.0, 0.0])  # near-duplicate
    different = np.array([[0.0, 1.0, 0.0]], dtype="float32")
    chunk_embeddings = np.vstack([similar, different])

    index = build_index(chunk_embeddings)
    query_emb = np.array([1.0, 0.1, 0.0], dtype="float32")

    results = mmr_search(index, query_emb, chunk_embeddings, top_k=2, fetch_k=4, lambda_mult=0.5)
    indices = [r[0] for r in results]
    # With low lambda, should pick one similar + the different one
    assert 3 in indices  # the different vector should appear


def test_mmr_search_relevance():
    """Most relevant result should still appear first."""
    chunk_embeddings = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ], dtype="float32")
    index = build_index(chunk_embeddings)
    query_emb = np.array([1.0, 0.0], dtype="float32")

    results = mmr_search(index, query_emb, chunk_embeddings, top_k=2, fetch_k=3, lambda_mult=0.7)
    assert results[0][0] == 0  # most relevant first


# --- BM25 tests ---


def test_bm25_search_keyword_match():
    chunks = [
        {"text": "Jose Rizal was a national hero", "metadata": {"page": 1}},
        {"text": "The weather in Manila is tropical", "metadata": {"page": 2}},
        {"text": "Rizal wrote Noli Me Tangere", "metadata": {"page": 3}},
    ]
    results = bm25_search(chunks, "Rizal", top_k=2)
    indices = [r[0] for r in results]
    assert 0 in indices
    assert 2 in indices
    assert 1 not in indices


def test_bm25_search_no_match():
    chunks = [
        {"text": "Hello world", "metadata": {"page": 1}},
    ]
    results = bm25_search(chunks, "zzzzz", top_k=1)
    assert results == []


# --- Merge tests ---


def test_merge_results_combines():
    mmr_results = [(0, 0.9), (1, 0.8)]
    bm25_results = [(2, 0.7), (0, 0.6)]
    merged = merge_results(mmr_results, bm25_results)
    indices = [r[0] for r in merged]
    assert 0 in indices
    assert 1 in indices
    assert 2 in indices


def test_merge_results_deduplicates():
    mmr_results = [(0, 0.9), (1, 0.8)]
    bm25_results = [(0, 0.7), (1, 0.6)]
    merged = merge_results(mmr_results, bm25_results)
    indices = [r[0] for r in merged]
    assert len(indices) == len(set(indices))


# --- Rerank tests ---


def test_rerank_reorders():
    chunks = [
        {"text": "The cat sat on the mat", "metadata": {"page": 1}},
        {"text": "Machine learning uses neural networks for classification", "metadata": {"page": 2}},
        {"text": "Deep learning is a subset of machine learning", "metadata": {"page": 3}},
    ]
    results = rerank(chunks, "What is deep learning?", top_k=3)
    assert len(results) == 3
    # Deep learning chunk should rank higher after reranking
    indices = [r[0] for r in results]
    dl_rank = indices.index(2)
    cat_rank = indices.index(0)
    assert dl_rank < cat_rank


# --- Full pipeline test ---


def test_search_pipeline_end_to_end():
    chunks = [
        {"text": "The EDSA Revolution happened in 1986", "metadata": {"page": 1}},
        {"text": "Manila is the capital of the Philippines", "metadata": {"page": 2}},
        {"text": "The People Power Revolution overthrew Marcos", "metadata": {"page": 3}},
        {"text": "Python is a programming language", "metadata": {"page": 4}},
    ]
    texts = [c["text"] for c in chunks]
    embeddings = get_embeddings(texts)
    index = build_index(embeddings)
    query = "EDSA Revolution"
    query_emb = get_embeddings([query])[0]

    results = search(
        index, query_emb, chunks=chunks,
        chunk_embeddings=embeddings, query_text=query, top_k=3,
    )
    assert len(results) <= 3
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    # EDSA-related chunks should be in results
    indices = [r[0] for r in results]
    assert 0 in indices or 2 in indices

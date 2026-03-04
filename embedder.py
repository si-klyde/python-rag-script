import hashlib
import logging
import os

import numpy as np
import faiss
from rank_bm25 import BM25Plus
from config import (
    USE_OPENAI, OPENAI_API_KEY, OPENAI_EMBED_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP,
    MMR_LAMBDA, MMR_FETCH_K_MULT, DENSE_WEIGHT, BM25_WEIGHT,
)

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")

_local_model = None
_ranker = None


def _cache_key(pdf_path):
    model_name = OPENAI_EMBED_MODEL if USE_OPENAI else "all-MiniLM-L6-v2"
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        h.update(f.read())
    h.update(f"{CHUNK_SIZE}:{CHUNK_OVERLAP}:{model_name}".encode())
    return h.hexdigest()


def load_cached(pdf_path):
    try:
        key = _cache_key(pdf_path)
    except FileNotFoundError:
        return None
    path = os.path.join(CACHE_DIR, f"{key}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    chunks = data["chunks"].tolist()
    embeddings = data["embeddings"]
    return chunks, embeddings


def save_cache(pdf_path, chunks, embeddings):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = _cache_key(pdf_path)
    path = os.path.join(CACHE_DIR, f"{key}.npz")
    np.savez(path, chunks=np.array(chunks, dtype=object), embeddings=embeddings)


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_model


def get_embeddings(texts, batch_size=256):
    if not texts:
        return np.empty((0, 0), dtype="float32")
    if isinstance(texts[0], dict):
        texts = [t["text"] for t in texts]

    if USE_OPENAI:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(input=batch, model=OPENAI_EMBED_MODEL)
            all_embeddings.extend(item.embedding for item in response.data)
        return np.array(all_embeddings, dtype="float32")

    model = _get_local_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings, dtype="float32")


def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def _cosine_sim(a, b):
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


def mmr_search(index, query_embedding, chunk_embeddings, top_k=5, fetch_k=15, lambda_mult=0.7):
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    fetch_k = min(fetch_k, index.ntotal)
    _, indices = index.search(query_embedding, fetch_k)
    candidate_ids = indices[0].tolist()

    candidates = chunk_embeddings[candidate_ids]
    query_sim = _cosine_sim(query_embedding[0], candidates).flatten()

    selected = []
    selected_embs = []
    remaining = list(range(len(candidate_ids)))

    for _ in range(min(top_k, len(candidate_ids))):
        if not remaining:
            break
        if not selected_embs:
            best = max(remaining, key=lambda i: query_sim[i])
        else:
            sel_matrix = np.vstack(selected_embs)
            remaining_arr = np.array(remaining)
            remaining_embs = candidates[remaining_arr]
            sims = _cosine_sim(remaining_embs, sel_matrix)
            redundancies = sims.max(axis=1)
            mmr_scores = lambda_mult * query_sim[remaining_arr] - (1 - lambda_mult) * redundancies
            best = remaining[int(np.argmax(mmr_scores))]

        selected.append(candidate_ids[best])
        selected_embs.append(candidates[best])
        remaining.remove(best)

    return [(idx, float(query_sim[candidate_ids.index(idx)])) for idx in selected]


_bm25_index = None
_bm25_chunks_id = None


def _get_bm25(chunks):
    global _bm25_index, _bm25_chunks_id
    if _bm25_index is None or id(chunks) != _bm25_chunks_id:
        tokenized = [c["text"].lower().split() for c in chunks]
        _bm25_index = BM25Plus(tokenized)
        _bm25_chunks_id = id(chunks)
    return _bm25_index


def bm25_search(chunks, query, top_k=5):
    bm25 = _get_bm25(chunks)
    scores = bm25.get_scores(query.lower().split())
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


def merge_results(dense_results, sparse_results, dense_weight=None, sparse_weight=None, k=60):
    dense_weight = DENSE_WEIGHT if dense_weight is None else dense_weight
    sparse_weight = BM25_WEIGHT if sparse_weight is None else sparse_weight
    scores = {}
    for rank, (idx, _) in enumerate(dense_results):
        scores[idx] = scores.get(idx, 0) + dense_weight / (k + rank + 1)
    for rank, (idx, _) in enumerate(sparse_results):
        scores[idx] = scores.get(idx, 0) + sparse_weight / (k + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(idx, score) for idx, score in ranked]


def _get_ranker():
    global _ranker
    if _ranker is None:
        from flashrank import Ranker
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return _ranker


def rerank(chunks, query, top_k=5, indices=None):
    from flashrank import RerankRequest
    ranker = _get_ranker()
    passages = [{"id": i, "text": chunks[i]["text"]} for i in range(len(chunks))]
    request = RerankRequest(query=query, passages=passages)
    ranked = ranker.rerank(request)
    results = []
    for r in ranked[:top_k]:
        pos = int(r["id"])
        idx = indices[pos] if indices else pos
        results.append((idx, float(r["score"])))
    return results


def search(index, query_embedding, chunks, chunk_embeddings,
           query_text, top_k=5):
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    fetch_k = min(top_k * MMR_FETCH_K_MULT, index.ntotal)

    # Step 1: MMR for diverse dense results
    mmr_results = mmr_search(
        index, query_embedding[0], chunk_embeddings,
        top_k=top_k, fetch_k=fetch_k, lambda_mult=MMR_LAMBDA,
    )

    # Step 2: BM25 for keyword results
    bm25_results = bm25_search(chunks, query_text, top_k=top_k)

    # Step 3: Reciprocal rank fusion
    merged = merge_results(mmr_results, bm25_results)

    # Step 4: Rerank with cross-encoder
    candidate_indices = [idx for idx, _ in merged[:top_k * 2]]
    candidate_chunks = [chunks[i] for i in candidate_indices]

    if candidate_chunks:
        reranked = rerank(candidate_chunks, query_text, top_k=top_k, indices=candidate_indices)
        return reranked

    return merged[:top_k]

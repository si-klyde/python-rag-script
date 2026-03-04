from unittest.mock import patch, MagicMock
from pdf_loader import load_and_chunk
from embedder import get_embeddings, build_index, search
from query_intent import classify_intent
from llm import ask


def test_pipeline_end_to_end_mocked():
    chunks = [
        {"text": "The EDSA Revolution happened in 1986.", "metadata": {"page": 1}},
        {"text": "Manila is the capital.", "metadata": {"page": 2}},
    ]

    texts = [c["text"] for c in chunks]
    embeddings = get_embeddings(texts)
    index = build_index(embeddings)

    query = "When did EDSA happen?"
    intent = classify_intent(query)
    assert intent == "factual"

    query_emb = get_embeddings([query])
    results = search(
        index, query_emb[0], chunks=chunks,
        chunk_embeddings=embeddings, query_text=query, top_k=1,
    )

    enriched = [
        {
            "text": chunks[idx]["text"],
            "score": score,
            "index": idx,
            "page": chunks[idx]["metadata"]["page"],
        }
        for idx, score in results
        if idx < len(chunks)
    ]

    assert any("EDSA" in c["text"] for c in enriched)

    with patch("ollama.chat") as mock_chat:
        mock_chat.return_value = {"message": {"content": "1986"}}
        with patch("llm.USE_OPENAI", False):
            response = ask(query, enriched, intent=intent)
            assert "1986" in response

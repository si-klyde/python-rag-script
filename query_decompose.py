import re
from config import DECOMPOSE_ENABLED, DECOMPOSE_MAX_SUB
from llm import _llm_chat

_DECOMPOSE_SYSTEM = (
    "You are a query decomposition assistant. "
    "Break the user's complex question into 2-4 simpler, independent sub-questions "
    "that together cover all facets of the original question. "
    "Output ONLY a numbered list. No preamble, no explanation."
)


def should_decompose(query: str) -> bool:
    if not DECOMPOSE_ENABLED:
        return False
    return len(query.split()) > 6


def decompose(query: str) -> list[str]:
    try:
        raw = _llm_chat(_DECOMPOSE_SYSTEM, query, temperature=0.3)
    except Exception:
        return [query]

    lines = raw.strip().splitlines()
    cleaned = []
    for line in lines:
        line = re.sub(r"^\s*[\d]+[.)]\s*", "", line)
        line = re.sub(r"^\s*[-*]\s*", "", line)
        line = line.strip()
        if line:
            cleaned.append(line)

    if not cleaned:
        return [query]

    return cleaned[:DECOMPOSE_MAX_SUB]


def multi_query_merge(result_lists: list[list[tuple[int, float]]], top_k: int, k: int = 60) -> list[tuple[int, float]]:
    if not result_lists:
        return []

    scores: dict[int, float] = {}
    for results in result_lists:
        for rank, (idx, _) in enumerate(results):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(idx, score) for idx, score in ranked[:top_k]]

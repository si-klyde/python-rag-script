from unittest.mock import patch
from query_decompose import should_decompose, decompose, multi_query_merge


# --- should_decompose ---

@patch("query_decompose.DECOMPOSE_ENABLED", False)
def test_should_decompose_disabled():
    assert should_decompose("complex multi-part question about many things") is False


def test_should_decompose_short():
    assert should_decompose("What year?") is False
    assert should_decompose("Who is Rizal?") is False
    assert should_decompose("Compare Spain and America") is False
    assert should_decompose("Why was Rizal important?") is False


def test_should_decompose_long():
    assert should_decompose(
        "Compare the economic and political impacts of colonialism on Philippine agriculture and trade",
    ) is True
    assert should_decompose(
        "What are the differences between Spanish and American colonial education systems",
    ) is True
    assert should_decompose(
        "What were the key reforms demanded by the Propaganda Movement members",
    ) is True


# --- decompose ---

@patch("query_decompose._llm_chat")
def test_decompose_parses_numbered_list(mock_chat):
    mock_chat.return_value = (
        "1. What was the economic impact?\n"
        "2. What was the political impact?\n"
        "3. How did it affect trade?"
    )
    result = decompose("complex question")
    assert len(result) == 3
    assert result[0] == "What was the economic impact?"
    assert result[1] == "What was the political impact?"


@patch("query_decompose._llm_chat")
def test_decompose_strips_dash_bullets(mock_chat):
    mock_chat.return_value = (
        "- What was the economic impact?\n"
        "- What was the political impact?"
    )
    result = decompose("complex question")
    assert len(result) == 2
    assert result[0] == "What was the economic impact?"


@patch("query_decompose._llm_chat")
def test_decompose_caps_at_max(mock_chat):
    mock_chat.return_value = "\n".join(f"{i+1}. Question {i+1}?" for i in range(10))
    with patch("query_decompose.DECOMPOSE_MAX_SUB", 4):
        result = decompose("complex question")
    assert len(result) == 4


@patch("query_decompose._llm_chat")
def test_decompose_filters_empties(mock_chat):
    mock_chat.return_value = "1. Real question?\n\n2. \n3. Another question?"
    result = decompose("complex question")
    assert len(result) == 2


@patch("query_decompose._llm_chat", side_effect=Exception("LLM down"))
def test_decompose_fallback_on_error(mock_chat):
    result = decompose("original question")
    assert result == ["original question"]


# --- multi_query_merge ---

def test_multi_query_merge_deduplicates():
    list1 = [(0, 0.9), (1, 0.8), (2, 0.7)]
    list2 = [(1, 0.85), (3, 0.75), (0, 0.6)]
    merged = multi_query_merge([list1, list2], top_k=4)
    ids = [idx for idx, _ in merged]
    assert len(ids) == len(set(ids))  # no dupes


def test_multi_query_merge_ranking():
    list1 = [(0, 0.9), (1, 0.8)]
    list2 = [(0, 0.85), (2, 0.75)]
    merged = multi_query_merge([list1, list2], top_k=3)
    # idx 0 appears in both lists → should rank highest
    assert merged[0][0] == 0


def test_multi_query_merge_respects_top_k():
    list1 = [(i, 0.9 - i * 0.1) for i in range(10)]
    merged = multi_query_merge([list1], top_k=5)
    assert len(merged) == 5


def test_multi_query_merge_empty_lists():
    merged = multi_query_merge([], top_k=5)
    assert merged == []


def test_multi_query_merge_single_list():
    list1 = [(0, 0.9), (1, 0.8), (2, 0.7)]
    merged = multi_query_merge([list1], top_k=3)
    assert len(merged) == 3
    assert merged[0][0] == 0  # highest ranked stays first

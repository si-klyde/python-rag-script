import pytest
from query_intent import classify_intent


@pytest.mark.parametrize("query,expected", [
    ("What is the capital of the Philippines?", "factual"),
    ("When did EDSA happen?", "factual"),
    ("Who was Jose Rizal?", "factual"),
    ("Compare Spanish and American colonization", "comparative"),
    ("What is the difference between Katipunan and Propaganda Movement?", "comparative"),
    ("How does Manila compare to Cebu?", "comparative"),
    ("similarities between Bonifacio and Rizal", "comparative"),
    ("Timeline of Philippine independence", "timeline"),
    ("What happened before the revolution?", "timeline"),
    ("events leading up to EDSA", "timeline"),
    ("after the war what changed?", "timeline"),
    ("Why was Rizal important?", "analytical"),
    ("Explain the impact of colonization", "analytical"),
    ("What caused the revolution?", "analytical"),
    ("How did trade influence culture?", "analytical"),
])
def test_classify_intent(query, expected):
    assert classify_intent(query) == expected


def test_classify_empty_string():
    assert classify_intent("") == "factual"


def test_classify_mixed_signals():
    # comparative keywords take priority over timeline
    result = classify_intent("Compare the timeline of Spanish and American rule")
    assert result == "comparative"

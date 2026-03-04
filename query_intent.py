import re

PATTERNS = {
    "comparative": re.compile(
        r"\b(compar\w*|differ\w*|distinction|versus|vs\.?|similarities|contrast)\b",
        re.IGNORECASE,
    ),
    "timeline": re.compile(
        r"\b(timeline|before|after|leading up|sequence|chronolog|events?\s+(?:of|in|during|leading))\b",
        re.IGNORECASE,
    ),
    "analytical": re.compile(
        r"\b(why|explain|cause[ds]?|impact|significance|import(?:ant|ance)|effect|influence|analyz|reason)\b",
        re.IGNORECASE,
    ),
}

PRIORITY = ["comparative", "timeline", "analytical"]


def classify_intent(query: str) -> str:
    for intent in PRIORITY:
        if PATTERNS[intent].search(query):
            return intent
    return "factual"

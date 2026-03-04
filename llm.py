from typing import Iterator
from config import (
    USE_OPENAI, OPENAI_API_KEY, OLLAMA_MODEL,
    OPENAI_CHAT_MODEL, LLM_TEMPERATURE,
)

_BASE_RULES = (
    "Rules:\n"
    "- Base your answer on the provided sources. You may infer and reason from them, but do not introduce outside facts.\n"
    "- Cite claims inline using [1], [2], etc. Include page numbers when available (e.g. [1, p.5]).\n"
    "- If the sources don't fully answer the question, state what's missing.\n"
    "- NEVER fabricate information that can't be derived from the sources.\n"
    "- Be concise — prefer short, direct answers over lengthy explanations.\n"
    "- If sources conflict, note the discrepancy."
)

SYSTEM_PROMPTS = {
    "factual": (
        "You are a document Q&A specialist. Answer questions based on the provided source excerpts. "
        "You may draw logical inferences from the sources, but do not introduce external knowledge.\n\n"
        f"{_BASE_RULES}"
    ),
    "comparative": (
        "You are a document Q&A specialist focused on comparison. "
        "Compare and contrast information using ONLY the provided source excerpts.\n\n"
        f"{_BASE_RULES}\n\n"
        "Comparison guidance:\n"
        "- Organize by dimensions of comparison, not by source.\n"
        "- Use parallel structure across compared items.\n"
        "- State similarities before differences."
    ),
    "timeline": (
        "You are a document Q&A specialist focused on chronology. "
        "Answer timeline and sequence questions using ONLY the provided source excerpts.\n\n"
        f"{_BASE_RULES}\n\n"
        "Timeline guidance:\n"
        "- Present events in chronological order.\n"
        "- Lead with dates or time markers when available.\n"
        "- Note gaps in the timeline explicitly."
    ),
    "analytical": (
        "You are a document Q&A specialist focused on analysis. "
        "Provide analytical explanations using ONLY the provided source excerpts.\n\n"
        f"{_BASE_RULES}\n\n"
        "Analysis guidance:\n"
        "- State the claim first, then supporting evidence, then reasoning.\n"
        "- Distinguish between what sources state and what can be inferred.\n"
        "- If evidence is partial, say so explicitly."
    ),
}


def get_system_prompt(intent: str) -> str:
    return SYSTEM_PROMPTS.get(intent, SYSTEM_PROMPTS["factual"])


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        page = chunk.get("page", "")
        page_attr = f' page="{page}"' if page else ""
        parts.append(
            f'<source id="{i}"{page_attr} relevance="{chunk["score"]:.2f}">\n'
            f'{chunk["text"]}\n'
            f'</source>'
        )
    return "\n".join(parts)


def build_prompt(query: str, chunks: list[dict], sub_questions: list[str] | None = None) -> str:
    context = build_context(chunks)
    prompt = f"<sources>\n{context}\n</sources>\n\n"

    if sub_questions:
        subs = "\n".join(f"- {q}" for q in sub_questions)
        prompt += f"<decomposition>\n{subs}\n</decomposition>\n\n"

    prompt += f"Question: {query}"
    return prompt


# --- LLM dispatch ---

def _llm_chat(system: str, user: str, temperature: float | None = None) -> str:
    temp = temperature if temperature is not None else LLM_TEMPERATURE

    if USE_OPENAI:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            temperature=temp,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content

    try:
        import ollama
        response = ollama.chat(
            model=OLLAMA_MODEL,
            options={"temperature": temp},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response["message"]["content"]
    except Exception as e:
        if "ConnectError" in type(e).__name__ or "ConnectionError" in type(e).__name__:
            raise ConnectionError(
                f"Cannot connect to Ollama. Ensure it's running: ollama serve\n"
                f"Then pull the model: ollama pull {OLLAMA_MODEL}"
            ) from e
        raise


def _llm_chat_stream(system: str, user: str, temperature: float | None = None) -> Iterator[str]:
    temp = temperature if temperature is not None else LLM_TEMPERATURE

    if USE_OPENAI:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        stream = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            temperature=temp,
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
        return

    try:
        import ollama
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            options={"temperature": temp},
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        for chunk in stream:
            content = chunk["message"]["content"]
            if content:
                yield content
    except Exception as e:
        if "ConnectError" in type(e).__name__ or "ConnectionError" in type(e).__name__:
            raise ConnectionError(
                f"Cannot connect to Ollama. Ensure it's running: ollama serve\n"
                f"Then pull the model: ollama pull {OLLAMA_MODEL}"
            ) from e
        raise


def ask(query: str, chunks: list[dict], intent: str = "factual") -> str:
    user_content = build_prompt(query, chunks)
    system_prompt = get_system_prompt(intent)
    return _llm_chat(system_prompt, user_content)


def ask_stream(
    query: str,
    chunks: list[dict],
    intent: str = "factual",
    sub_questions: list[str] | None = None,
) -> Iterator[str]:
    user_content = build_prompt(query, chunks, sub_questions=sub_questions)
    system_prompt = get_system_prompt(intent)
    yield from _llm_chat_stream(system_prompt, user_content)

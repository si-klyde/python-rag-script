from unittest.mock import patch, MagicMock
from llm import (
    build_prompt, build_context, get_system_prompt, ask, SYSTEM_PROMPTS,
    _llm_chat, _llm_chat_stream, ask_stream,
)


def _enriched_chunks():
    return [
        {"text": "Chunk one.", "score": 0.95, "index": 0, "page": 5},
        {"text": "Chunk two.", "score": 0.80, "index": 1, "page": 12},
    ]


def test_build_context_xml_tags():
    chunks = _enriched_chunks()
    ctx = build_context(chunks)
    assert '<source id="1"' in ctx
    assert '<source id="2"' in ctx
    assert "</source>" in ctx
    assert "Chunk one." in ctx
    assert "Chunk two." in ctx


def test_build_context_shows_relevance():
    chunks = _enriched_chunks()
    ctx = build_context(chunks)
    assert 'relevance="0.95"' in ctx
    assert 'relevance="0.80"' in ctx


def test_build_context_shows_page():
    chunks = _enriched_chunks()
    ctx = build_context(chunks)
    assert 'page="5"' in ctx
    assert 'page="12"' in ctx


def test_build_context_no_page_when_missing():
    chunks = [{"text": "No page.", "score": 0.9, "index": 0}]
    ctx = build_context(chunks)
    assert "page=" not in ctx


def test_build_prompt_uses_xml_sources():
    chunks = _enriched_chunks()
    prompt = build_prompt("What happened?", chunks)
    assert "<sources>" in prompt
    assert "</sources>" in prompt
    assert "Chunk one." in prompt
    assert "Chunk two." in prompt
    assert "Question: What happened?" in prompt


def test_build_prompt_no_redundant_instructions():
    chunks = _enriched_chunks()
    prompt = build_prompt("What happened?", chunks)
    assert "Answer based only" not in prompt
    assert "Cite sources" not in prompt


def test_get_system_prompt_per_intent():
    for intent in ["factual", "comparative", "timeline", "analytical"]:
        prompt = get_system_prompt(intent)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "NEVER" in prompt
        assert "provided source" in prompt.lower() or "provided sources" in prompt.lower()


def test_get_system_prompt_has_base_rules():
    for intent in SYSTEM_PROMPTS:
        prompt = SYSTEM_PROMPTS[intent]
        assert "Cite claims inline" in prompt
        assert "NEVER fabricate" in prompt
        assert "do not introduce" in prompt.lower()


def test_get_system_prompt_has_page_citation_rule():
    for intent in SYSTEM_PROMPTS:
        prompt = SYSTEM_PROMPTS[intent]
        assert "page numbers" in prompt


def test_get_system_prompt_intent_specific():
    assert "dimensions of comparison" in SYSTEM_PROMPTS["comparative"]
    assert "chronological order" in SYSTEM_PROMPTS["timeline"]
    assert "claim first, then supporting evidence" in SYSTEM_PROMPTS["analytical"]


def test_get_system_prompt_fallback():
    prompt = get_system_prompt("unknown_intent")
    assert prompt == SYSTEM_PROMPTS["factual"]


@patch("llm.USE_OPENAI", False)
@patch("ollama.chat")
def test_ask_ollama(mock_chat):
    mock_chat.return_value = {"message": {"content": "Test answer"}}
    chunks = _enriched_chunks()
    result = ask("question?", chunks, intent="factual")
    assert result == "Test answer"
    call_args = mock_chat.call_args
    messages = call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


@patch("llm.USE_OPENAI", True)
@patch("llm.OpenAI", create=True)
def test_ask_openai(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_choice = MagicMock()
    mock_choice.message.content = "OpenAI answer"
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    chunks = _enriched_chunks()
    with patch("llm.OPENAI_API_KEY", "fake-key"):
        with patch("openai.OpenAI", return_value=mock_client):
            result = ask("question?", chunks, intent="analytical")
            assert result == "OpenAI answer"


@patch("llm.USE_OPENAI", False)
@patch("ollama.chat")
def test_ask_uses_intent_system_prompt(mock_chat):
    mock_chat.return_value = {"message": {"content": "answer"}}
    chunks = _enriched_chunks()
    ask("question?", chunks, intent="comparative")
    messages = mock_chat.call_args.kwargs["messages"]
    system_msg = messages[0]["content"]
    assert system_msg == SYSTEM_PROMPTS["comparative"]


# --- _llm_chat ---

@patch("llm.USE_OPENAI", False)
@patch("ollama.chat")
def test_llm_chat_ollama(mock_chat):
    mock_chat.return_value = {"message": {"content": "ollama reply"}}
    result = _llm_chat("system msg", "user msg")
    assert result == "ollama reply"
    call_args = mock_chat.call_args
    assert call_args.kwargs["messages"][0]["content"] == "system msg"
    assert call_args.kwargs["messages"][1]["content"] == "user msg"


@patch("llm.USE_OPENAI", True)
@patch("llm.OPENAI_API_KEY", "fake-key")
def test_llm_chat_openai():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "openai reply"
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    with patch("openai.OpenAI", return_value=mock_client):
        result = _llm_chat("system msg", "user msg")
    assert result == "openai reply"


# --- _llm_chat_stream ---

@patch("llm.USE_OPENAI", False)
@patch("ollama.chat")
def test_llm_chat_stream_ollama(mock_chat):
    mock_chat.return_value = iter([
        {"message": {"content": "hello"}},
        {"message": {"content": " world"}},
    ])
    tokens = list(_llm_chat_stream("sys", "usr"))
    assert tokens == ["hello", " world"]
    assert mock_chat.call_args.kwargs["stream"] is True


@patch("llm.USE_OPENAI", True)
@patch("llm.OPENAI_API_KEY", "fake-key")
def test_llm_chat_stream_openai():
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "hello"
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = " world"
    chunk3 = MagicMock()
    chunk3.choices = [MagicMock()]
    chunk3.choices[0].delta.content = None

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])

    with patch("openai.OpenAI", return_value=mock_client):
        tokens = list(_llm_chat_stream("sys", "usr"))
    assert tokens == ["hello", " world"]
    assert mock_client.chat.completions.create.call_args.kwargs["stream"] is True


# --- build_prompt with sub_questions ---

def test_build_prompt_without_sub_questions():
    chunks = _enriched_chunks()
    prompt = build_prompt("Main question?", chunks)
    assert "<decomposition>" not in prompt


def test_build_prompt_with_sub_questions():
    chunks = _enriched_chunks()
    subs = ["Sub Q1?", "Sub Q2?"]
    prompt = build_prompt("Main question?", chunks, sub_questions=subs)
    assert "<decomposition>" in prompt
    assert "Sub Q1?" in prompt
    assert "Sub Q2?" in prompt
    assert "</decomposition>" in prompt


# --- ask_stream ---

@patch("llm.USE_OPENAI", False)
@patch("ollama.chat")
def test_ask_stream_yields_tokens(mock_chat):
    mock_chat.return_value = iter([
        {"message": {"content": "Token1"}},
        {"message": {"content": " Token2"}},
    ])
    chunks = _enriched_chunks()
    tokens = list(ask_stream("question?", chunks, intent="factual"))
    assert tokens == ["Token1", " Token2"]


@patch("llm.USE_OPENAI", False)
@patch("ollama.chat")
def test_ask_stream_with_sub_questions(mock_chat):
    mock_chat.return_value = iter([
        {"message": {"content": "answer"}},
    ])
    chunks = _enriched_chunks()
    subs = ["Sub Q1?"]
    tokens = list(ask_stream("question?", chunks, intent="factual", sub_questions=subs))
    assert tokens == ["answer"]
    user_msg = mock_chat.call_args.kwargs["messages"][1]["content"]
    assert "<decomposition>" in user_msg

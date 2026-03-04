import pytest
from pdf_loader import extract_markdown, chunk_documents, load_and_chunk
from config import CHUNK_SIZE, PDF_PATH


def test_extract_markdown_returns_pages():
    pages = extract_markdown(PDF_PATH)
    assert isinstance(pages, list)
    assert len(pages) > 0
    for page in pages:
        assert isinstance(page, dict)
        assert "text" in page
        assert "metadata" in page
        assert isinstance(page["text"], str)


def test_extract_markdown_has_page_numbers():
    pages = extract_markdown(PDF_PATH)
    for page in pages:
        assert "page" in page["metadata"]
        assert isinstance(page["metadata"]["page"], int)
        assert page["metadata"]["page"] >= 1


def test_extract_markdown_invalid_path():
    with pytest.raises(Exception):
        extract_markdown("nonexistent.pdf")


def test_chunk_documents_preserves_metadata():
    pages = [
        {"text": "# Heading\n\nSome content here. " * 50, "metadata": {"page": 1}},
        {"text": "# Another\n\nMore content here. " * 50, "metadata": {"page": 2}},
    ]
    chunks = chunk_documents(pages)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "text" in chunk
        assert "metadata" in chunk
        assert "page" in chunk["metadata"]
        assert chunk["metadata"]["page"] in [1, 2]


def test_chunk_documents_markdown_separators():
    text = "# Section One\n\nParagraph one. " * 30 + "\n## Section Two\n\nParagraph two. " * 30
    pages = [{"text": text, "metadata": {"page": 1}}]
    chunks = chunk_documents(pages)
    assert len(chunks) > 1


def test_chunk_documents_empty():
    chunks = chunk_documents([])
    assert chunks == []


def test_chunk_documents_empty_text():
    pages = [{"text": "", "metadata": {"page": 1}}]
    chunks = chunk_documents(pages)
    assert chunks == []


def test_load_and_chunk_returns_dicts():
    chunks = load_and_chunk(PDF_PATH)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, dict)
        assert "text" in chunk
        assert "metadata" in chunk
        assert isinstance(chunk["text"], str)
        assert len(chunk["text"]) > 0

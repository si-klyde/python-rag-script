import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

MARKDOWN_SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", " ", ""]


def extract_markdown(pdf_path):
    pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    return [
        {"text": page["text"], "metadata": {"page": page["metadata"]["page"] + 1}}
        for page in pages
        if page["text"].strip()
    ]


def chunk_documents(pages):
    if not pages:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=MARKDOWN_SEPARATORS,
        length_function=len,
    )
    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for text in splits:
            chunks.append({"text": text, "metadata": dict(page["metadata"])})
    return chunks


def load_and_chunk(pdf_path):
    pages = extract_markdown(pdf_path)
    return chunk_documents(pages)

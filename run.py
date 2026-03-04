import sys
import os
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live

from config import PDF_PATH, TOP_K
from pdf_loader import load_and_chunk
from embedder import get_embeddings, build_index, search, load_cached, save_cache
from query_intent import classify_intent
from llm import ask_stream
from query_decompose import should_decompose, decompose, multi_query_merge

console = Console()


def _load_data():
    with console.status("Loading cached embeddings..."):
        cached = load_cached(PDF_PATH)
    if cached:
        chunks, chunk_embeddings = cached
        console.print(f"Loaded [bold]{len(chunks)}[/] chunks from cache")
    else:
        with console.status("Loading and chunking PDF..."):
            chunks = load_and_chunk(PDF_PATH)
        console.print(f"Created [bold]{len(chunks)}[/] chunks from {PDF_PATH}")

        with console.status("Generating embeddings..."):
            chunk_embeddings = get_embeddings(chunks)
        save_cache(PDF_PATH, chunks, chunk_embeddings)

    with console.status("Building search index..."):
        index = build_index(chunk_embeddings)
    return chunks, chunk_embeddings, index


def _handle_query(query, chunks, chunk_embeddings, index):
    console.print(Panel(query, title="Query", border_style="blue"))

    intent = classify_intent(query)
    console.print(f"Detected intent: [bold magenta]{intent}[/]")

    # Query decomposition
    sub_questions = None
    if should_decompose(query):
        with console.status("Decomposing query..."):
            raw_subs = decompose(query)
        # Normalize: fallback [query] → None
        if raw_subs and raw_subs != [query]:
            sub_questions = raw_subs
            subs_text = "\n".join(f"  {i}. {q}" for i, q in enumerate(sub_questions, 1))
            console.print(Panel(subs_text, title="Sub-questions", border_style="yellow"))

    # Search: original + sub-queries
    with console.status("Searching..."):
        query_embedding = get_embeddings([query])
        all_result_lists = []

        original_results = search(
            index, query_embedding[0], chunks=chunks,
            chunk_embeddings=chunk_embeddings, query_text=query, top_k=TOP_K,
        )
        all_result_lists.append(original_results)

        if sub_questions:
            for sq in sub_questions:
                sq_embedding = get_embeddings([sq])
                sq_results = search(
                    index, sq_embedding[0], chunks=chunks,
                    chunk_embeddings=chunk_embeddings, query_text=sq, top_k=TOP_K,
                )
                all_result_lists.append(sq_results)

    # Merge results
    used_merge = len(all_result_lists) > 1
    if used_merge:
        final_results = multi_query_merge(all_result_lists, top_k=TOP_K)
    else:
        final_results = original_results

    enriched = []
    for idx, score in final_results:
        chunk = chunks[idx]
        page = chunk["metadata"].get("page", "?")
        enriched.append({
            "text": chunk["text"],
            "score": score,
            "index": idx,
            "page": page,
        })

    console.print()
    score_label = "rank" if used_merge else "relevance"
    for i, chunk in enumerate(enriched, 1):
        console.print(Panel(
            chunk["text"],
            title=f"[{i}] page {chunk['page']} | {score_label}: {chunk['score']:.2f}",
            border_style="dim",
        ))

    # Streaming LLM response
    text = ""
    with Live(
        Panel(Markdown(""), title="Response", border_style="green"),
        console=console, refresh_per_second=4,
    ) as live:
        for token in ask_stream(query, enriched, intent=intent, sub_questions=sub_questions):
            text += token
            live.update(Panel(Markdown(text), title="Response", border_style="green"))


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        chunks, chunk_embeddings, index = _load_data()
        _handle_query(query, chunks, chunk_embeddings, index)
        return

    chunks, chunk_embeddings, index = _load_data()
    console.print(Panel(
        f"[bold]{PDF_PATH}[/] — {len(chunks)} chunks indexed",
        title="RAG Ready",
        border_style="cyan",
    ))

    try:
        while True:
            console.print()
            query = console.input("[bold cyan]Query > [/]").strip()
            if not query or query.lower() in ("quit", "exit"):
                break
            console.rule()
            _handle_query(query, chunks, chunk_embeddings, index)
    except (KeyboardInterrupt, EOFError):
        pass

    console.print("\n[dim]Goodbye[/]")


if __name__ == "__main__":
    main()

import os
import glob
from dotenv import load_dotenv

load_dotenv()

USE_OPENAI = bool(os.environ.get("TEAMIFIED_OPENAI_API_KEY"))
OPENAI_API_KEY = os.environ.get("TEAMIFIED_OPENAI_API_KEY", "")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 10

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_EMBED_MODEL = "text-embedding-3-small"

LLM_TEMPERATURE = 0.3

DECOMPOSE_ENABLED = True
DECOMPOSE_MAX_SUB = 4

MMR_LAMBDA = 0.7
MMR_FETCH_K_MULT = 3
DENSE_WEIGHT = 0.6
BM25_WEIGHT = 0.4


def detect_pdf_path():
    explicit = os.environ.get("PDF_PATH")
    if explicit:
        return explicit
    root = os.path.dirname(__file__)
    preferred = os.path.join(root, "philippine_history.pdf")
    if os.path.exists(preferred):
        return preferred
    pdfs = glob.glob(os.path.join(root, "*.pdf"))
    if pdfs:
        return pdfs[0]
    raise FileNotFoundError("No PDF found. Set PDF_PATH in .env or place philippine_history.pdf in the project root.")


PDF_PATH = detect_pdf_path()

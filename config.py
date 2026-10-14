import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
INDEXES_DIR = DATA_DIR / "indexes"
REPORTS_DIR = DATA_DIR / "reports"
DB_PATH = DATA_DIR / "detector.db"

# Ensure data directories exist
for path in [DATA_DIR, UPLOADS_DIR, INDEXES_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env, with .env.example fallback
env_file = BASE_DIR / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    load_dotenv(dotenv_path=BASE_DIR / ".env.example")

# LLM Providers & Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("DEFAULT_OLLAMA_MODEL", "gemma3:latest")
FALLBACK_OLLAMA_MODEL = "llama2:latest"

# Optional Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DEFAULT_GEMINI_MODEL = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-3.8-flash")

# Embedding Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384

# Document Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "140"))  # tokens / words
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "3"))
MIN_RELEVANCE_THRESHOLD = float(os.getenv("MIN_RELEVANCE_THRESHOLD", "0.35"))

# NLI Model Configuration
NLI_MODEL_NAME = os.getenv("NLI_MODEL_NAME", "roberta-large-mnli")
NLI_DEVICE = "cuda" if os.getenv("USE_CUDA", "false").lower() == "true" else "cpu"

# Verification Verdict Labels
VERDICT_SUPPORTED = "Supported"
VERDICT_CONTRADICTED = "Contradicted"
VERDICT_UNSUPPORTED = "Unsupported"

# Decision Thresholds
ENTAILMENT_THRESHOLD = float(os.getenv("ENTAILMENT_THRESHOLD", "0.50"))
CONTRADICTION_THRESHOLD = float(os.getenv("CONTRADICTION_THRESHOLD", "0.50"))
AMBIGUITY_THRESHOLD = float(os.getenv("AMBIGUITY_THRESHOLD", "0.15"))

# UI Theme & Styling constants
THEME_NAVY = "#0F172A"
THEME_CARD_BG = "#1E293B"
COLOR_SUPPORTED = "#10B981"    # Green
COLOR_CONTRADICTED = "#EF4444" # Red
COLOR_UNSUPPORTED = "#F59E0B"  # Amber/Yellow

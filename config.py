"""Configuration settings and environment variables for Travel Policy Agent."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# LLM & Embedding settings
# Supported providers: "groq" (free fast API), "ollama" (free local offline), "google" (free tier API), "huggingface_local" (free local CPU), "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Model Names
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GOOGLE_MODEL_NAME = os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-flash")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
HF_LOCAL_MODEL_NAME = os.getenv("HF_LOCAL_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

# Embeddings (Default is 100% free local CPU sentence-transformers)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
HUGGINGFACE_EMBEDDING_MODEL = os.getenv(
    "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "").strip()


# RAG & Graph Hyperparameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "4"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

# LangSmith / LangChain Tracing Setup
if (
    os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    and LANGCHAIN_API_KEY
    and LANGCHAIN_API_KEY != "your-langsmith-api-key-here"
):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "travel-policy-agent"))
else:
    # Disable tracing if no valid key to avoid noisy runtime warnings
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


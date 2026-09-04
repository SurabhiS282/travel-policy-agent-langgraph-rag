"""LLM and Embeddings Factory with Free Models Support.

Supports:
1. Groq (100% Free API, ultra-fast Llama-3.3-70B / Llama-3.1-8B)
2. Ollama (100% Free local offline execution, e.g. Llama-3.2, Qwen-2.5)
3. Google Gemini (Free tier API, Gemini-1.5-Flash / Gemini-2.0-Flash)
4. HuggingFace Local Pipeline (100% Free local CPU execution)
5. OpenAI (Optional paid GPT-4o-mini)
6. HuggingFace Embeddings (100% Free local CPU sentence-transformers)
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from src.config import (
    LLM_PROVIDER,
    GROQ_MODEL_NAME,
    GROQ_API_KEY,
    OLLAMA_MODEL_NAME,
    OLLAMA_BASE_URL,
    GOOGLE_MODEL_NAME,
    GOOGLE_API_KEY,
    OPENAI_MODEL_NAME,
    OPENAI_API_KEY,
    HF_LOCAL_MODEL_NAME,
    EMBEDDING_PROVIDER,
    HUGGINGFACE_EMBEDDING_MODEL,
)


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    """Returns an initialized Chat Model instance based on configuration."""
    provider = LLM_PROVIDER.lower()

    if provider == "groq":
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set in .env.\n"
                "Get a free API key at: https://console.groq.com/keys\n"
                "Or set LLM_PROVIDER='ollama' to run 100% locally with Ollama without any API key."
            )
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=GROQ_MODEL_NAME,
                temperature=temperature,
                groq_api_key=GROQ_API_KEY,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq LLM: {e}")

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=OLLAMA_MODEL_NAME,
                base_url=OLLAMA_BASE_URL,
                temperature=temperature,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to local Ollama instance at {OLLAMA_BASE_URL}: {e}\n"
                "Make sure Ollama is installed and running (`ollama run llama3.2`)."
            )

    elif provider == "google":
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is not set in .env.\n"
                "Get a free Gemini API key at: https://aistudio.google.com/app/apikey"
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=GOOGLE_MODEL_NAME,
                temperature=temperature,
                google_api_key=GOOGLE_API_KEY,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Google Gemini LLM: {e}")

    elif provider == "huggingface_local":
        try:
            from transformers import pipeline
            from langchain_huggingface import HuggingFacePipeline
            from langchain_core.language_models.chat_models import BaseChatModel

            pipe = pipeline(
                "text-generation",
                model=HF_LOCAL_MODEL_NAME,
                max_new_tokens=512,
                temperature=temperature if temperature > 0 else 0.1,
                do_sample=temperature > 0,
            )
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize local HuggingFace pipeline: {e}")

    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in .env.")
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=OPENAI_MODEL_NAME,
                temperature=temperature,
                api_key=OPENAI_API_KEY,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI LLM: {e}")

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{provider}'. "
            "Supported free options: 'groq', 'ollama', 'google', 'huggingface_local'."
        )


def get_embeddings() -> Embeddings:
    """Returns an initialized Embeddings instance.
    
    Defaults to local HuggingFace embeddings on CPU (sentence-transformers/all-MiniLM-L6-v2)
    which is 100% free, runs offline, and requires NO API key.
    """
    provider = EMBEDDING_PROVIDER.lower()

    if provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=HUGGINGFACE_EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HuggingFace embeddings: {e}")

    elif provider == "google":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required for Google embeddings.")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY,
        )

    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY,
        )

    else:
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: '{provider}'.")

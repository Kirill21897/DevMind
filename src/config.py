import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # Ollama Settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://192.168.88.21:91/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3-vl:8b")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    
    # Paths
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    DOCS_SOURCE_PATH: str = os.getenv("DOCS_SOURCE_PATH", "./data/knowledge_base")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./output")
    EVALUATION_LOG_FILE: str = os.getenv("EVALUATION_LOG_FILE", "data/evaluation/ragas_dataset.jsonl")
    
    # Models
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # LangFuse Settings
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    @property
    def native_ollama_url(self) -> str:
        """Returns the base URL without /v1 suffix for native Ollama clients."""
        if self.OLLAMA_BASE_URL.endswith("/v1"):
            return self.OLLAMA_BASE_URL[:-3]
        return self.OLLAMA_BASE_URL

config = Config()

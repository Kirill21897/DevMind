import logging
import sys
import ollama
from .config import config

def setup_logger(name: str = "DevMind") -> logging.Logger:
    """Configures and returns a standard logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Console Handler
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.INFO)
        c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        
        logger.addHandler(c_handler)
    return logger

# Global client instance to reuse connection and avoid ResourceWarning
_ollama_client = None

def get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = ollama.Client(host=config.native_ollama_url)
    return _ollama_client

def get_ollama_embedding(text: str, model: str = config.EMBEDDING_MODEL) -> list[float]:
    """
    Wrapper for Ollama embeddings to be used across the project.
    """
    try:
        client = get_ollama_client()
        response = client.embeddings(model=model, prompt=text)
        return response.get("embedding", [])
    except Exception as e:
        logger = logging.getLogger("DevMind")
        logger.error(f"Error getting embedding: {e}")
        return []

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """
    Splits text into chunks with overlap.
    """
    chunks = []
    if not text:
        return []
        
    start = 0
    if len(text) <= chunk_size:
        return [text]
        
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

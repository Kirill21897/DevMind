
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from langchain_ollama import ChatOllama, OllamaEmbeddings
from src.config import config

llm = ChatOllama(model=config.LLM_MODEL, base_url=config.native_ollama_url)
embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.native_ollama_url)

try:
    cp = ContextPrecision()
    cp.llm = llm # Patching llm if needed
    print("ContextPrecision init ok")
except Exception as e:
    print(f"ContextPrecision init failed: {e}")

try:
    f = Faithfulness()
    f.llm = llm
    print("Faithfulness init ok")
except Exception as e:
    print(f"Faithfulness init failed: {e}")

try:
    ar = AnswerRelevancy()
    ar.llm = llm
    ar.embeddings = embeddings
    print("AnswerRelevancy init ok")
except Exception as e:
    print(f"AnswerRelevancy init failed: {e}")

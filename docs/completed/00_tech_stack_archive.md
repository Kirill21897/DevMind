# 02. Технический стек и Спецификация инструментов

## 1. Технический стек (Environment)

### Core Dependencies:
*   **Python**: 3.10+
*   **Orchestration**: `Simple Agent Pattern` (Native Python).
*   **LLM Provider**: **Ollama** (Local Inference).
    *   *Model*: `llama3` или `qwen2.5-coder` (для кода).
*   **Embeddings**: **Ollama** (`nomic-embed-text` или `mxbai-embed-large`).
*   **Vector Database**: `chromadb` (Persistent Client).
*   **Reranker**: `sentence-transformers` (Cross-Encoder) для улучшения релевантности поиска.
*   **Search**: `duckduckgo-search`.
*   **Evaluation**: `ragas` + `datasets`.

### Configuration (`.env`):
```ini
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3:latest
EMBEDDING_MODEL=nomic-embed-text
CHROMA_DB_PATH=./data/chroma_db
DOCS_SOURCE_PATH=./data/knowledge_base
```

### Requirements (`requirements.txt`):
```text
openai>=1.0.0          # Клиент совместим с Ollama API
chromadb>=0.4.0
duckduckgo-search>=5.0.0
ragas>=0.1.0
sentence-transformers  # Для Reranker
torch                  # Для Reranker
pandas
python-dotenv
colorama
tiktoken
```

## 2. Спецификация Инструментов (Detailed Design)

### Tool 1: `KnowledgeRetriever` (RAG + Rerank)
Этот инструмент отвечает за **умный поиск** по базе знаний.

*   **Logic**:
    1.  **Retrieval**: Получает `top_k=10` кандидатов из ChromaDB (по косинусному сходству).
    2.  **Reranking**:
        *   Загружает модель Cross-Encoder (например, `cross-encoder/ms-marco-MiniLM-L-6-v2`).
        *   Оценивает пары `(query, document_chunk)`.
        *   Сортирует по скору и берет `top_n=3`.
    3.  **Output**: Возвращает только самые релевантные чанки.
    
    *Почему это важно*: Векторный поиск часто находит "похожие слова", но не "ответы". Reranker исправляет это, понимая контекст вопроса.

### Tool 2: `WebExplorer` (Ground Truth Augmentation)
*   **Logic**:
    1.  Использует `DDGS().text(keywords, max_results=3)`.
    2.  Возвращает список сниппетов (Title + URL + Body).

### Tool 3: `SolutionReport` (Artifact Generator)
*   **Logic**:
    1.  Принимает: `filename` и `content`.
    2.  Записывает файл в папку `output/`.

## 3. Подготовка к Evaluation (Ragas Integration)

### Класс `RagasTracker`:
Логирует взаимодействия для последующей оценки.

```python
class RagasTracker:
    # ... (Singleton pattern) ...
    
    def log_interaction(self, query: str, response: str, used_contexts: list[str]):
        """
        used_contexts: Сюда должны попадать чанки ПОСЛЕ Reranking'а.
        """
        entry = {
            "question": query,
            "answer": response,
            "contexts": used_contexts,
            "ground_truth": "" 
        }
        save_to_jsonl(entry, "ragas_dataset.jsonl")
```

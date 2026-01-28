# Фаза 2: Слой данных и Индексация (Data Ingestion)

В этой фазе мы реализуем механизм превращения Markdown-документации в векторный индекс ChromaDB с использованием Ollama Embeddings.

## 2.1. Концепция Ingestion Pipeline
Мы не будем встраивать индексацию в самого агента (чтобы не замедлять старт). Вместо этого создадим отдельный скрипт `ingest.py`.

**Алгоритм:**
1.  **Load**: Рекурсивный обход папки `data/knowledge_base`.
2.  **Split**: Разбиение текста на чанки (Chunking). Оптимальный размер для RAG кода — 500-1000 токенов с перекрытием.
3.  **Embed**: Отправка текста в Ollama (`nomic-embed-text`) -> получение вектора.
4.  **Store**: Сохранение (Вектор + Текст + Метаданные) в ChromaDB.

## 2.2. Детальная реализация `ingest.py`

Этот файл должен находиться в корне проекта.

### Необходимые импорты
```python
import os
import chromadb
from chromadb.config import Settings
import ollama
from dotenv import load_dotenv
import glob
from tqdm import tqdm
```

### Шаг 1: Подключение к Ollama Embeddings
Нужно написать функцию-обертку, так как ChromaDB нативно может требовать специфический класс, но проще делать вручную:

```python
def get_ollama_embedding(text, model="nomic-embed-text"):
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]
```

### Шаг 2: Чанкинг (Text Splitting)
Реализуйте функцию `split_text`, которая делит текст по заголовкам или просто по символам.
*Рекомендация*: Для простоты начните с разбиения по `\n## ` (заголовки markdown) или фиксированному размеру символов (1000 chars) с overlap (100 chars).

```python
def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
```

### Шаг 3: Основной цикл (Main Loop)
```python
def ingest_documents():
    # 1. Инициализация DB
    client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH"))
    collection = client.get_or_create_collection(name="devmind_docs")

    # 2. Поиск файлов
    docs_path = os.getenv("DOCS_SOURCE_PATH")
    files = glob.glob(f"{docs_path}/**/*.md", recursive=True)

    print(f"Found {len(files)} documents.")

    # 3. Обработка
    for file_path in tqdm(files):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        chunks = chunk_text(content)
        
        # Подготовка батча для вставки
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            vector = get_ollama_embedding(chunk, model=os.getenv("EMBEDDING_MODEL"))
            
            ids.append(f"{os.path.basename(file_path)}_{i}")
            embeddings.append(vector)
            documents.append(chunk)
            metadatas.append({"source": file_path, "chunk_id": i})
            
        # 4. Вставка в БД
        if ids:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
    print("Ingestion complete!")
```

## 2.3. Проверка результата
Для проверки создайте временный скрипт `test_db.py`:
```python
import chromadb
client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_collection("devmind_docs")
print(f"Total documents: {collection.count()}")
print(collection.peek()) # Показать первые элементы
```

## 2.4. Критерии завершения фазы
- [ ] Скрипт `ingest.py` написан.
- [ ] Папка `data/knowledge_base` содержит хотя бы один тестовый .md файл.
- [ ] Запуск `python ingest.py` проходит без ошибок.
- [ ] В папке `data/chroma_db` появились файлы базы данных (sqlite3, bin).

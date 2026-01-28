# Фаза 3: Реализация Инструментов (Tools Development)

В этой фазе мы реализуем "руки" агента: поиск с реранкингом, веб-поиск и запись файлов. Все функции должны быть типизированы и документированы, чтобы агент мог корректно их вызывать.

## 3.1. Структура `src/tools.py`

Этот файл будет содержать класс `ToolSet`, который инициализирует необходимые ресурсы (подключение к БД, загрузка модели Reranker) при старте.

### Инициализация
```python
from sentence_transformers import CrossEncoder
import chromadb
from duckduckgo_search import DDGS
import os

class ToolSet:
    def __init__(self):
        # 1. ChromaDB Client
        self.chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH"))
        self.collection = self.chroma_client.get_collection("devmind_docs")
        
        # 2. Reranker Model (загружается один раз)
        print("Loading Reranker model...")
        self.reranker = CrossEncoder(os.getenv("RERANKER_MODEL"))
        print("Reranker loaded.")
```

## 3.2. Инструмент 1: `retrieve_knowledge` (RAG + Rerank)

Это самый сложный инструмент. Он реализует двухступенчатый поиск.

**Алгоритм:**
1.  **Embed Query**: Получить вектор вопроса через Ollama (функция из Фазы 2).
2.  **Vector Search**: Запросить топ-10 (`n_results=10`) из ChromaDB.
3.  **Cross-Encoding**: Создать пары `[query, doc_text]` для всех 10 кандидатов.
4.  **Rerank**: Прогнать пары через `self.reranker.predict()`.
5.  **Sort & Filter**: Отсортировать по скору (от большего к меньшему) и вернуть топ-3.

**Сигнатура:**
```python
def retrieve_knowledge(self, query: str) -> str:
    """
    Ищет информацию в локальной базе знаний. Использует Reranker для повышения точности.
    Возвращает релевантные фрагменты текста.
    """
    # ... implementation ...
    return formatted_string_of_top_3_chunks
```

## 3.3. Инструмент 2: `web_search`

Обертка над DuckDuckGo.

**Алгоритм:**
1.  Вызвать `DDGS().text(query, max_results=3)`.
2.  Склеить Title, URL и Body в одну строку.
3.  Обработать ошибки (try/except), если интернет недоступен.

**Сигнатура:**
```python
def web_search(self, query: str) -> str:
    """
    Ищет информацию в интернете (Google/DuckDuckGo).
    Использовать, если retrieve_knowledge не дал результатов.
    """
    # ... implementation ...
```

## 3.4. Инструмент 3: `save_solution`

Инструмент для создания артефактов.

**Алгоритм:**
1.  Сформировать полный путь: `os.path.join(os.getenv("OUTPUT_DIR"), filename)`.
2.  Записать `content` в файл.
3.  Вернуть сообщение об успехе: "File saved: ..."

**Сигнатура:**
```python
def save_solution(self, filename: str, content: str) -> str:
    """
    Сохраняет решение (код, инструкцию) в файл markdown.
    Filename должен быть без путей, только имя (например, 'guide.md').
    """
    # ... implementation ...
```

## 3.5. Определение схем инструментов (JSON Schema)

Для того чтобы Ollama (или OpenAI Client) понимал, как вызывать эти функции, нужно описать их схемы. Добавьте в `src/tools.py` переменную `TOOLS_SCHEMA`:

```python
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge",
            "description": "Search local knowledge base for technical details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    # ... добавить схемы для web_search и save_solution ...
]
```

## 3.6. Критерии завершения фазы
- [ ] Файл `src/tools.py` создан.
- [ ] Класс `ToolSet` инициализируется (модель Reranker качается).
- [ ] Метод `retrieve_knowledge` возвращает отсортированные данные.
- [ ] `web_search` возвращает реальные результаты из интернета.
- [ ] `save_solution` создает файл на диске.

# DevMind AI Агент

DevMind — это продвинутый локальный ИИ-ассистент, разработанный специально для программистов и инди-разработчиков. Он использует локальные LLM (через Ollama), технологию RAG (Retrieval-Augmented Generation) и автономные агентные возможности для помощи в написании кода, работе с документацией и поиске информации.

## Возможности

-   **Локальный Интеллект**: Работает на базе Ollama (поддерживает модели `qwen2.5-coder`, `llama3` и другие).
-   **RAG Пайплайн**:
    -   Двухэтапный поиск: Векторный поиск (ChromaDB) + Переранжирование (Cross-Encoder).
    -   Индексация локальной документации в формате Markdown.
-   **Автономные Инструменты**:
    -   `retrieve_knowledge`: Поиск по локальной базе знаний.
    -   `web_search`: Поиск в интернете (через DuckDuckGo).
    -   `save_solution`: Сохранение сгенерированного кода и гайдов в файлы.
-   **Современный UI**: Красивый веб-интерфейс на базе [Chainlit](https://chainlit.io/).
-   **Наблюдаемость (Observability)**: Полный трекинг и отладка через [LangFuse](https://langfuse.com/).
-   **Оценка Качества**: Встроенный пайплайн оценки Ragas для проверки точности контекста и релевантности ответов.

## Структура Проекта

```
├── app.py                  # Точка входа веб-приложения (Chainlit)
├── main.py                 # Точка входа CLI (командная строка)
├── src/
│   ├── agent.py            # Логика агента (ReAct цикл)
│   ├── config.py           # Настройки конфигурации
│   ├── tools.py            # Инструменты агента
│   └── utils.py            # Утилиты
├── scripts/
│   ├── ingest_data.py      # Скрипт индексации документов в ChromaDB
│   └── evaluate_rag.py     # Скрипт оценки качества (Ragas)
├── docs/
│   └── completed/          # История разработки и документация
└── requirements.txt        # Зависимости Python
```

## Установка и Настройка

### Требования

1.  **Python 3.10+**
2.  **Ollama**: Установите и запустите [Ollama](https://ollama.com/).
    -   Скачайте необходимые модели:
        ```bash
        ollama pull qwen2.5-coder:14b  # Или ваша предпочтительная модель
        ollama pull nomic-embed-text   # Для эмбеддингов
        ```

### Установка

1.  Клонируйте репозиторий.
2.  Создайте виртуальное окружение:
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```
3.  Установите зависимости:
    ```bash
    pip install -r requirements.txt
    ```

### Конфигурация

Создайте файл `.env` (необязательно), чтобы переопределить настройки по умолчанию:

```env
# Настройки LLM
LLM_MODEL=qwen2.5-coder:14b
EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434/v1

# LangFuse (Опционально)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Использование

### 1. Веб-интерфейс (Рекомендуется)

Запустите Chainlit UI:

```bash
chainlit run app.py -w
```

Чат будет доступен по адресу `http://localhost:8000`. Флаг `-w` включает автоперезагрузку при изменении кода.

### 2. Командная строка (CLI)

Выполнить одиночный запрос:

```bash
python main.py --query "Как реализовать RAG пайплайн?"
```

Запустить интерактивную сессию:

```bash
python main.py
```

### 3. Обновление Базы Знаний

Чтобы добавить новые `.md` файлы из папки `docs/` в базу знаний:

```bash
python scripts/ingest_data.py
```

### 4. Оценка Качества (Evaluation)

Для запуска оценки Ragas на основе собранных логов:

```bash
python scripts/evaluate_rag.py
```

## Лицензия

MIT

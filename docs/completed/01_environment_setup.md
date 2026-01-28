# Фаза 1: Подготовка окружения и Настройка Ollama

Этот документ содержит исчерпывающие инструкции по настройке рабочей среды.

## 1.1. Структура Проекта
Создайте следующую структуру каталогов и файлов. Она спроектирована для разделения данных, кода и артефактов.

```text
D:\Projects\AI_Research_Assistant_for_Indie_Developers\
├── .env                    # Переменные окружения (Ключи, настройки)
├── .gitignore              # Исключения git
├── requirements.txt        # Зависимости Python
├── main.py                 # Точка входа (CLI)
├── ingest.py               # Скрипт индексации базы знаний
├── evaluate.py             # Скрипт запуска оценки Ragas
├── data/
│   ├── knowledge_base/     # Исходные MD файлы (ВАШИ ДОКУМЕНТЫ)
│   ├── chroma_db/          # Хранилище векторной БД (генерируется авто)
│   └── evaluation/         # Датасеты и отчеты Ragas
├── src/
│   ├── __init__.py
│   ├── agent.py            # Логика агента (Orchestrator)
│   ├── tools.py            # Реализация инструментов (RAG, Web, File)
│   └── tracker.py          # Логирование для Ragas
└── output/                 # Папка для сгенерированных решений
```

## 1.2. Настройка Python окружения

1.  **Создание виртуального окружения**:
    ```powershell
    python -m venv venv
    ```

2.  **Активация**:
    ```powershell
    .\venv\Scripts\Activate
    ```

3.  **Файл `requirements.txt`**:
    Создайте файл с точными версиями библиотек для стабильности:
    ```text
    # Core AI
    openai>=1.10.0           # Используем OpenAI Client для работы с Ollama API
    chromadb>=0.4.22         # Векторная БД
    duckduckgo-search>=4.4   # Поиск
    
    # Reranking & Torch
    sentence-transformers>=2.3.1
    torch>=2.2.0             # CPU версия ок, если есть GPU - pytorch-cuda
    
    # Evaluation
    ragas>=0.1.0
    datasets>=2.17.0
    
    # Utils
    python-dotenv>=1.0.1
    pandas>=2.2.0
    colorama>=0.4.6
    tiktoken>=0.6.0
    tqdm>=4.66.0             # Прогресс бары
    ```

4.  **Установка**:
    ```powershell
    pip install -r requirements.txt
    ```

## 1.3. Настройка и Проверка Ollama

Проект полностью полагается на локальные модели.

1.  **Установка Ollama**:
    Убедитесь, что Ollama установлена (скачать с [ollama.com](https://ollama.com)).

2.  **Загрузка Моделей**:
    Нам нужны две модели: одна для генерации текста (умная), вторая для эмбеддингов (векторов).
    ```powershell
    # LLM (Мозг агента)
    ollama pull llama3:latest 
    # Альтернатива для кодинга: ollama pull qwen2.5-coder:7b
    
    # Embedding Model (Векторизация)
    ollama pull nomic-embed-text
    ```

3.  **Проверка работоспособности**:
    Выполните curl-запрос, чтобы убедиться, что API доступен:
    ```powershell
    curl http://localhost:11434/api/tags
    ```
    *Ожидаемый результат*: JSON со списком моделей.

## 1.4. Конфигурация (.env)

Создайте файл `.env` в корне проекта.

```ini
# --- Ollama Configuration ---
# Базовый URL API Ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
# ВНИМАНИЕ: Для OpenAI Client используем /v1 суффикс, для нативных запросов - без него.
# В коде будем использовать base_url="http://localhost:11434/v1"

# Имена моделей (должны совпадать с тем, что вы скачали)
LLM_MODEL=llama3:latest
EMBEDDING_MODEL=nomic-embed-text

# --- Project Paths ---
CHROMA_DB_PATH=./data/chroma_db
DOCS_SOURCE_PATH=./data/knowledge_base
OUTPUT_DIR=./output

# --- Reranker Config ---
# Модель для Cross-Encoder (будет скачана huggingface-hub автоматически при первом запуске)
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

## 1.5. Критерии завершения фазы
- [ ] Структура папок создана.
- [ ] `pip install` прошел без ошибок.
- [ ] Команда `ollama list` показывает `llama3` и `nomic-embed-text`.
- [ ] Файл `.env` создан и заполнен.

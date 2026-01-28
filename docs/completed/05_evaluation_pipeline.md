# Фаза 5: Оценка и Метрики (Evaluation Pipeline)

Эта фаза посвящена анализу качества работы агента с использованием библиотеки Ragas. Мы будем использовать собранный датасет `ragas_dataset.jsonl`.

## 5.1. Подготовка Ragas к работе с Ollama

По умолчанию Ragas использует OpenAI. Нам нужно переключить его на Ollama через LangChain обертки.

### Скрипт `evaluate.py`

```python
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    faithfulness,
    answer_relevance,
)
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv

# Настройка моделей для Ragas (Evaluator Models)
# Для оценки лучше использовать сильную модель (например, llama3 или mistral)
def get_evaluator_models():
    llm = ChatOllama(model="llama3:latest", base_url=os.getenv("OLLAMA_BASE_URL"))
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=os.getenv("OLLAMA_BASE_URL"))
    return llm, embeddings

def run_evaluation():
    load_dotenv()
    
    # 1. Загрузка данных
    df = pd.read_json("data/evaluation/ragas_dataset.jsonl", lines=True)
    
    # Преобразуем в формат HuggingFace Dataset
    # Ragas ожидает колонки: question, answer, contexts, ground_truth (опционально)
    dataset = Dataset.from_pandas(df)
    
    # 2. Инициализация моделей
    eval_llm, eval_embeddings = get_evaluator_models()
    
    # 3. Запуск оценки
    print("Starting evaluation... This may take time.")
    results = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevance,
        ],
        llm=eval_llm,
        embeddings=eval_embeddings
    )
    
    # 4. Вывод результатов
    print("\nEvaluation Results:")
    print(results)
    
    # 5. Сохранение отчета
    df_results = results.to_pandas()
    df_results.to_csv("data/evaluation/report.csv", index=False)
    print("Report saved to data/evaluation/report.csv")

if __name__ == "__main__":
    run_evaluation()
```

## 5.2. Интерпретация Метрик

*   **Faithfulness (Верность)**: Измеряет, насколько ответ агента соответствует найденному контексту. Низкий балл = галлюцинации.
*   **Answer Relevance (Релевантность ответа)**: Насколько ответ вообще относится к вопросу (независимо от правильности).
*   **Context Precision (Точность контекста)**: Насколько полезны были найденные документы (топ-3). Если Reranker работает плохо, эта метрика упадет.

## 5.3. Процесс улучшения (Iteration Loop)

1.  Запустите агента и задайте 5-10 тестовых вопросов.
2.  Убедитесь, что `ragas_dataset.jsonl` заполнился.
3.  Запустите `python evaluate.py`.
4.  Проанализируйте `report.csv`.
    *   Если `context_precision` низкий -> Проверьте Reranker или Embedding модель.
    *   Если `faithfulness` низкий -> Проверьте промпт агента (System Prompt).

## 5.4. Критерии завершения фазы
- [ ] Скрипт `evaluate.py` запускается.
- [ ] Ragas успешно подключается к Ollama.
- [ ] Генерируется CSV отчет с числами (0.0 - 1.0).

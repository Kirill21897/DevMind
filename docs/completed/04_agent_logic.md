# Фаза 4: Логика Агента и Ragas Logger

В этой фазе мы связываем все воедино: создаем цикл общения агента с пользователем и интегрируем сбор данных для метрик Ragas.

## 4.1. Логирование для Ragas (`src/tracker.py`)

Нам нужно сохранять данные в формате `jsonl`, где каждая строка — это JSON объект с полями `question`, `answer`, `contexts`.

```python
import json
import os
from datetime import datetime

class RagasTracker:
    def __init__(self, log_file="data/evaluation/ragas_dataset.jsonl"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_turn(self, question, answer, contexts):
        """
        question: str - запрос пользователя
        answer: str - финальный ответ агента
        contexts: list[str] - список текстов, которые вернули retrieve_knowledge/web_search
        """
        entry = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": "", # Оставляем пустым
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

## 4.2. Агент (`src/agent.py`)

Здесь реализуется `ReAct` (Reasoning + Acting) цикл с использованием OpenAI Client, настроенного на Ollama.

### Настройка Клиента
```python
from openai import OpenAI
import os
from .tools import ToolSet, TOOLS_SCHEMA
from .tracker import RagasTracker

class Agent:
    def __init__(self):
        self.client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL"),
            api_key="ollama" # required but unused
        )
        self.tools = ToolSet()
        self.tracker = RagasTracker()
        self.history = [{"role": "system", "content": "You are DevMind, an expert AI assistant..."}]
        
        # Буфер для контекстов текущей сессии (для Ragas)
        self.current_contexts = []
```

### Основной цикл (Run Loop)
```python
    def run(self, user_query):
        self.history.append({"role": "user", "content": user_query})
        self.current_contexts = [] # Сброс контекстов для нового вопроса
        
        while True:
            # 1. Запрос к LLM
            response = self.client.chat.completions.create(
                model=os.getenv("LLM_MODEL"),
                messages=self.history,
                tools=TOOLS_SCHEMA,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            # 2. Если LLM хочет вызвать инструмент
            if message.tool_calls:
                self.history.append(message) # Добавляем "мысль" агента в историю
                
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    
                    # Вызов функции
                    result = self._execute_tool(func_name, args)
                    
                    # Ragas Hook: Если это был поиск, сохраняем контекст
                    if func_name in ["retrieve_knowledge", "web_search"]:
                         # Предполагаем, что result это текст или список
                         self.current_contexts.append(str(result))
                    
                    # Добавляем результат в историю
                    self.history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": str(result)
                    })
            else:
                # 3. Финальный ответ
                final_answer = message.content
                self.history.append({"role": "assistant", "content": final_answer})
                
                # Логируем в Ragas Tracker
                self.tracker.log_turn(user_query, final_answer, self.current_contexts)
                
                return final_answer
```

### Диспетчер инструментов `_execute_tool`
```python
    def _execute_tool(self, name, args):
        if name == "retrieve_knowledge":
            return self.tools.retrieve_knowledge(args["query"])
        elif name == "web_search":
            return self.tools.web_search(args["query"])
        elif name == "save_solution":
            return self.tools.save_solution(args["filename"], args["content"])
        else:
            return "Error: Unknown tool"
```

## 4.3. Точка входа (`main.py`)

Простой CLI интерфейс.

```python
from src.agent import Agent
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    agent = Agent()
    print("DevMind AI ready. Type 'exit' to quit.")
    
    while True:
        q = input("\nUser: ")
        if q.lower() in ["exit", "quit"]: break
        
        ans = agent.run(q)
        print(f"\nAgent: {ans}")
```

## 4.4. Критерии завершения фазы
- [ ] Агент корректно выбирает инструменты (видно в логах).
- [ ] Агент способен поддерживать диалог (Context window).
- [ ] После ответа в `data/evaluation/ragas_dataset.jsonl` появляется новая запись с непустым полем `contexts`.

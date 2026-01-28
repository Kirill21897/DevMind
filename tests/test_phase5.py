import unittest
import os
import shutil
import pandas as pd
import json
from unittest.mock import MagicMock, patch
import evaluate

class TestPhase5(unittest.TestCase):
    
    def setUp(self):
        self.test_data_dir = "data/test_evaluation"
        os.makedirs(self.test_data_dir, exist_ok=True)
        self.test_file = os.path.join(self.test_data_dir, "ragas_dataset.jsonl")
        
        # Создаем фиктивный датасет
        data = [
            {
                "question": "What is DevMind?",
                "answer": "DevMind is an AI agent.",
                "contexts": ["DevMind is an AI assistant for developers."],
                "ground_truth": "DevMind is an AI agent."
            }
        ]
        with open(self.test_file, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    def tearDown(self):
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)

    @patch("evaluate.evaluate")
    @patch("evaluate.ChatOllama")
    @patch("evaluate.OllamaEmbeddings")
    def test_evaluation_flow(self, mock_embed, mock_llm, mock_evaluate):
        """Тест потока оценки (с моками моделей и функции evaluate)."""
        
        # Мокаем результат evaluate
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = pd.DataFrame({
            "question": ["q"], "context_precision": [1.0]
        })
        mock_evaluate.return_value = mock_result
        
        # Подменяем путь к файлу в функции (через патч open не выйдет просто так, 
        # проще изменить глобальную переменную если бы она была, или передать аргумент.
        # Но так как путь захардкожен, мы временно подменим его через создание реального файла 
        # по нужному пути или запатчим pd.read_json)
        
        with patch("evaluate.pd.read_json") as mock_read:
            mock_read.return_value = pd.DataFrame([{
                "question": "q",
                "answer": "a",
                "contexts": ["c"],
                "ground_truth": "g"
            }])
            
            evaluate.run_evaluation()
            
            # Проверяем, что evaluate был вызван
            self.assertTrue(mock_evaluate.called)

if __name__ == "__main__":
    unittest.main()

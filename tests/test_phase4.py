import unittest
import os
import json
import shutil
from unittest.mock import MagicMock, patch
from src.agent import Agent
from src.tracker import RagasTracker

class TestPhase4(unittest.TestCase):
    
    def setUp(self):
        os.environ["OLLAMA_BASE_URL"] = "http://mock-url"
        os.environ["LLM_MODEL"] = "mock-model"
        self.test_log_file = "data/test_evaluation/dataset.jsonl"
        
    def tearDown(self):
        if os.path.exists(os.path.dirname(self.test_log_file)):
            shutil.rmtree(os.path.dirname(self.test_log_file))

    def test_tracker(self):
        """Проверка логирования Ragas."""
        tracker = RagasTracker(log_file=self.test_log_file)
        tracker.log_turn("Q", "A", ["C1", "C2"])
        
        self.assertTrue(os.path.exists(self.test_log_file))
        with open(self.test_log_file, "r") as f:
            line = f.readline()
            data = json.loads(line)
            self.assertEqual(data["question"], "Q")
            self.assertEqual(data["answer"], "A")
            self.assertEqual(data["contexts"], ["C1", "C2"])

    @patch("src.agent.OpenAI")
    @patch("src.agent.ToolSet")
    def test_agent_run_simple(self, mock_toolset, mock_openai):
        """Проверка простого ответа агента без инструментов."""
        # Мокаем ответ OpenAI
        mock_client = mock_openai.return_value
        mock_response = MagicMock()
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "Hello, User!"
        mock_client.chat.completions.create.return_value = mock_response
        
        agent = Agent()
        # Подменяем трекер на тестовый
        agent.tracker = RagasTracker(log_file=self.test_log_file)
        
        response = agent.run("Hi")
        self.assertEqual(response, "Hello, User!")
        
        # Проверяем что лог записался
        self.assertTrue(os.path.exists(self.test_log_file))

if __name__ == "__main__":
    unittest.main()

import unittest
import os
import shutil
from unittest.mock import MagicMock, patch
from src.tools import ToolSet

class TestPhase3(unittest.TestCase):
    
    def setUp(self):
        # Настройка переменных окружения для теста
        os.environ["CHROMA_DB_PATH"] = "./data/test_chroma_db_tools"
        os.environ["OUTPUT_DIR"] = "./data/test_output"
        os.environ["RERANKER_MODEL"] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
        os.makedirs(os.environ["OUTPUT_DIR"], exist_ok=True)
        
    def tearDown(self):
        if os.path.exists(os.environ["OUTPUT_DIR"]):
            shutil.rmtree(os.environ["OUTPUT_DIR"])
        # БД удаляем аккуратно
        if os.path.exists(os.environ["CHROMA_DB_PATH"]):
            shutil.rmtree(os.environ["CHROMA_DB_PATH"])

    @patch("src.tools.chromadb.PersistentClient")
    @patch("src.tools.CrossEncoder")
    def test_toolset_initialization(self, mock_reranker, mock_chroma):
        """Проверка инициализации ToolSet."""
        tools = ToolSet()
        self.assertIsNotNone(tools.chroma_client)
        self.assertIsNotNone(tools.reranker)

    @patch("src.tools.DDGS")
    def test_web_search(self, mock_ddgs):
        """Проверка веб-поиска."""
        # Мокаем ответ DDGS
        mock_ddgs_instance = mock_ddgs.return_value
        mock_ddgs_instance.text.return_value = [
            {"title": "Test Result", "href": "http://example.com", "body": "This is a test."}
        ]
        
        tools = ToolSet()
        result = tools.web_search("test query")
        self.assertIn("Test Result", result)
        self.assertIn("http://example.com", result)

    def test_save_solution(self):
        """Проверка сохранения файла."""
        # Патчим CrossEncoder и ChromaClient чтобы не грузить их реально
        with patch("src.tools.chromadb.PersistentClient"), \
             patch("src.tools.CrossEncoder"):
            
            tools = ToolSet()
            filename = "test_file.md"
            content = "# Hello World"
            
            result = tools.save_solution(filename, content)
            
            self.assertIn("saved successfully", result)
            
            expected_path = os.path.join(os.environ["OUTPUT_DIR"], filename)
            self.assertTrue(os.path.exists(expected_path))
            
            with open(expected_path, "r") as f:
                read_content = f.read()
            self.assertEqual(read_content, content)

if __name__ == "__main__":
    unittest.main()

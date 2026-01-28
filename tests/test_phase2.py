import unittest
import os
import shutil
import chromadb
from unittest.mock import patch, MagicMock
import ingest

class TestPhase2(unittest.TestCase):
    
    def setUp(self):
        self.test_db_path = "./data/test_chroma_db"
        self.test_docs_path = "./data/test_knowledge_base"
        
        # Создаем тестовые директории
        os.makedirs(self.test_db_path, exist_ok=True)
        os.makedirs(self.test_docs_path, exist_ok=True)
        
        # Создаем тестовый файл
        with open(os.path.join(self.test_docs_path, "test.md"), "w", encoding="utf-8") as f:
            f.write("# Test Title\nThis is a test content for chunking and embedding.")
            
        # Настраиваем окружение для теста
        os.environ["CHROMA_DB_PATH"] = self.test_db_path
        os.environ["DOCS_SOURCE_PATH"] = self.test_docs_path
        os.environ["EMBEDDING_MODEL"] = "nomic-embed-text"

    def tearDown(self):
        # Очистка после тестов
        if os.path.exists(self.test_db_path):
            shutil.rmtree(self.test_db_path)
        if os.path.exists(self.test_docs_path):
            shutil.rmtree(self.test_docs_path)

    def test_chunking(self):
        """Проверка функции разбиения текста."""
        text = "a" * 1500
        chunks = ingest.chunk_text(text, chunk_size=1000, overlap=100)
        self.assertTrue(len(chunks) >= 2)
        self.assertTrue(len(chunks[0]) == 1000)

    @patch("ingest.ollama.embeddings")
    def test_ingestion_mock(self, mock_embeddings):
        """Тест процесса индексации с моком Ollama."""
        # Мокаем ответ от Ollama
        mock_embeddings.return_value = {"embedding": [0.1] * 768}
        
        # Запускаем индексацию
        ingest.ingest_documents()
        
        # Проверяем результат в БД
        client = chromadb.PersistentClient(path=self.test_db_path)
        collection = client.get_collection("devmind_docs")
        count = collection.count()
        
        self.assertGreater(count, 0, "Database should not be empty after ingestion")
        
        # Проверяем метаданные
        data = collection.peek()
        self.assertIn("source", data["metadatas"][0])

if __name__ == "__main__":
    unittest.main()

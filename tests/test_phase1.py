import os
import sys
import unittest
import requests
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

class TestPhase1Setup(unittest.TestCase):
    def test_directories_exist(self):
        """Проверка наличия всех необходимых папок."""
        required_dirs = [
            "data/knowledge_base",
            "data/chroma_db",
            "data/evaluation",
            "src",
            "output",
            "tests"
        ]
        for d in required_dirs:
            self.assertTrue(os.path.exists(d), f"Directory missing: {d}")

    def test_env_variables(self):
        """Проверка наличия и загрузки переменных из .env."""
        required_vars = [
            "OLLAMA_BASE_URL",
            "LLM_MODEL",
            "EMBEDDING_MODEL",
            "CHROMA_DB_PATH",
            "DOCS_SOURCE_PATH",
            "OUTPUT_DIR",
            "RERANKER_MODEL"
        ]
        for var in required_vars:
            val = os.getenv(var)
            self.assertIsNotNone(val, f"Environment variable missing: {var}")
            self.assertNotEqual(val, "", f"Environment variable is empty: {var}")

    def test_imports(self):
        """Проверка, что ключевые библиотеки установлены и импортируются."""
        libs = [
            "chromadb",
            "openai",
            "duckduckgo_search",
            "ragas",
            "sentence_transformers",
            "torch"
        ]
        for lib in libs:
            with self.subTest(lib=lib):
                try:
                    print(f"Importing {lib}...", end="", flush=True)
                    __import__(lib)
                    print(" OK")
                except ImportError as e:
                    print(f" FAIL: {e}")
                    self.fail(f"Failed to import dependency: {lib} - {e}")

    def test_structure(self):
        """Проверка структуры проекта (файлы и папки)."""
        required_dirs = [
            "data/knowledge_base",
            "data/chroma_db",
            "data/evaluation",
            "src",
            "output"
        ]
        required_files = [
            "main.py",
            "ingest.py",
            "evaluate.py",
            "src/agent.py",
            "src/tools.py",
            "src/tracker.py",
            ".env",
            "requirements.txt"
        ]
        
        for d in required_dirs:
            with self.subTest(dir=d):
                self.assertTrue(os.path.isdir(d), f"Directory missing: {d}")
                
        for f in required_files:
            with self.subTest(file=f):
                self.assertTrue(os.path.isfile(f), f"File missing: {f}")

    def test_ollama_connection(self):
        """Проверка связи с Ollama API."""
        base_url = os.getenv("OLLAMA_BASE_URL")
        if base_url.endswith("/v1"):
            native_url = base_url[:-3]
        else:
            native_url = base_url
            
        url = f"{native_url}/api/tags"
        print(f"Testing Ollama connection at: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f" WARN: Ollama returned status {response.status_code}")
                self.skipTest(f"Ollama API returned status {response.status_code}")
                
            data = response.json()
            models = [m['name'] for m in data.get('models', [])]
            llm_model = os.getenv("LLM_MODEL")
            
            # Проверка модели (нестрогая)
            found_llm = any(llm_model.split(':')[0] in m for m in models)
            if not found_llm:
                print(f" WARN: Model {llm_model} not found. Available: {models}")
                # Не фейлим тест, если модели нет, так как её можно спуллить позже
            
        except requests.exceptions.RequestException as e:
            print(f" WARN: Could not connect to Ollama at {url}: {e}")
            self.skipTest("Ollama server is unreachable (skipping connection check)")

if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch, MagicMock
import os
from src.config import config
from src.utils import get_ollama_embedding

class TestLocalConfig(unittest.TestCase):
    def test_config_values(self):
        # Verify defaults (which should now match the user's input because we updated config.py)
        # Note: If .env is loaded, it overrides config.py defaults, but we set .env to same values.
        expected_base = "http://192.168.88.21:91/v1"
        self.assertEqual(config.OLLAMA_BASE_URL, expected_base)
        self.assertEqual(config.LLM_MODEL, "qwen3-vl:8b")
        self.assertEqual(config.native_ollama_url, "http://192.168.88.21:91")

    @patch("src.utils.ollama.Client")
    def test_embedding_uses_configured_host(self, mock_client_cls):
        # Setup mock
        mock_client_instance = MagicMock()
        mock_client_cls.return_value = mock_client_instance
        mock_client_instance.embeddings.return_value = {"embedding": [0.1, 0.2]}

        # Call function
        get_ollama_embedding("test")

        # Verify Client was initialized with correct host
        mock_client_cls.assert_called_with(host="http://192.168.88.21:91")
        
        # Verify embeddings was called
        mock_client_instance.embeddings.assert_called()

if __name__ == "__main__":
    unittest.main()

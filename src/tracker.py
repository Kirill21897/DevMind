import json
import os
from datetime import datetime
from src.config import config
from src.utils import setup_logger

logger = setup_logger("Tracker")

class RagasTracker:
    def __init__(self, log_file: str = config.EVALUATION_LOG_FILE):
        self.log_file = log_file
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory for logs: {e}")

    def log_turn(self, question: str, answer: str, contexts: list[str]):
        """
        Logs the interaction for Ragas evaluation.
        """
        entry = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": "", 
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Error logging to Ragas dataset: {e}")

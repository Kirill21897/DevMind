import os
import sys
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import config
from src.utils import setup_logger

logger = setup_logger("Evaluator")

def get_evaluator_models():
    # LangChain ChatOllama uses 'base_url'
    llm = ChatOllama(model=config.LLM_MODEL, base_url=config.native_ollama_url) 
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL, base_url=config.native_ollama_url)
    return llm, embeddings

def run_evaluation():
    input_file = config.EVALUATION_LOG_FILE
    if not os.path.exists(input_file):
        logger.error(f"Dataset file not found at {input_file}")
        return

    try:
        df = pd.read_json(input_file, lines=True)
        if df.empty:
            logger.warning("Dataset is empty.")
            return
    except ValueError as e:
        logger.error(f"Error reading dataset: {e}")
        return
    
    dataset = Dataset.from_pandas(df)
    
    logger.info("Loading evaluator models (Ollama)...")
    eval_llm, eval_embeddings = get_evaluator_models()
    
    logger.info("Starting evaluation... This may take time.")
    try:
        results = evaluate(
            dataset=dataset,
            metrics=[
                ContextPrecision(),
                Faithfulness(),
                AnswerRelevancy(),
            ],
            llm=eval_llm,
            embeddings=eval_embeddings
        )
        
        logger.info("\nEvaluation Results:")
        print(results)
        
        output_file = os.path.join(os.path.dirname(input_file), "report.csv")
        df_results = results.to_pandas()
        df_results.to_csv(output_file, index=False)
        logger.info(f"Report saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")

if __name__ == "__main__":
    run_evaluation()

import os
import sys
import glob
import chromadb
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import config
from src.utils import setup_logger, get_ollama_embedding, chunk_text

logger = setup_logger("Ingest")

def ingest_documents():
    logger.info(f"Connecting to ChromaDB at {config.CHROMA_DB_PATH}...")
    try:
        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        collection = client.get_or_create_collection(name="devmind_docs")
    except Exception as e:
        logger.critical(f"Failed to connect to ChromaDB: {e}")
        return

    logger.info(f"Scanning documents in {config.DOCS_SOURCE_PATH}...")
    files = glob.glob(f"{config.DOCS_SOURCE_PATH}/**/*.md", recursive=True)
    
    logger.info(f"Found {len(files)} documents.")
    
    for file_path in tqdm(files, desc="Processing files"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"Skipping empty file: {file_path}")
                continue
                
            chunks = chunk_text(content)
            
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                vector = get_ollama_embedding(chunk)
                if not vector:
                    logger.warning(f"Failed to get embedding for chunk {i} in {file_path}")
                    continue
                    
                # Unique ID
                file_name = os.path.basename(file_path)
                chunk_id = f"{file_name}_{i}"
                
                ids.append(chunk_id)
                embeddings.append(vector)
                documents.append(chunk)
                metadatas.append({"source": file_path, "chunk_index": i})
            
            if ids:
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            
    logger.info(f"Ingestion complete! Total documents in DB: {collection.count()}")

if __name__ == "__main__":
    ingest_documents()

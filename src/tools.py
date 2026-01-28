import os
import chromadb
from sentence_transformers import CrossEncoder
from ddgs import DDGS
from src.config import config
from src.utils import setup_logger, get_ollama_embedding

logger = setup_logger("Tools")

class ToolSet:
    def __init__(self):
        # 1. ChromaDB Client
        try:
            self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
            self.collection = self.chroma_client.get_or_create_collection(name="devmind_docs")
        except Exception as e:
            logger.error(f"Could not connect to ChromaDB: {e}")
            self.collection = None

        # 2. Reranker Model
        logger.info(f"Loading Reranker model: {config.RERANKER_MODEL}...")
        try:
            self.reranker = CrossEncoder(config.RERANKER_MODEL)
            logger.info("Reranker loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading Reranker: {e}")
            self.reranker = None
            
        # 3. Output directory
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    def retrieve_knowledge(self, query: str) -> str:
        """
        Search local knowledge base for technical details.
        """
        if not self.collection:
            return "Error: Database not initialized."

        # 1. Embed Query
        query_vector = get_ollama_embedding(query)
        if not query_vector:
            return "Error: Could not generate embedding for query."

        # 2. Vector Search
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=10
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return f"Error querying database: {e}"
        
        if not results['documents'] or not results['documents'][0]:
            return "No relevant information found in knowledge base."
            
        candidates = results['documents'][0]
        
        # Fallback if no reranker
        if not self.reranker:
            return "\n\n".join(candidates[:3])

        # 3. Cross-Encoding & Reranking
        pairs = [[query, doc] for doc in candidates]
        try:
            scores = self.reranker.predict(pairs)
            
            scored_candidates = sorted(
                zip(scores, candidates), 
                key=lambda x: x[0], 
                reverse=True
            )
            top_3 = [doc for score, doc in scored_candidates[:3]]
            return "\n\n---\n\n".join(top_3)
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return "\n\n".join(candidates[:3])

    def web_search(self, query: str) -> str:
        """
        Search the internet via DuckDuckGo.
        """
        try:
            # Using new ddgs package
            results = DDGS().text(query, max_results=5)
            if not results:
                return "No results found on the web."
                
            formatted_results = []
            for res in results:
                title = res.get('title', 'No Title')
                href = res.get('href', 'No URL')
                body = res.get('body', 'No Content')
                formatted_results.append(f"Title: {title}\nURL: {href}\nContent: {body}")
                
            return "\n\n---\n\n".join(formatted_results)
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Error performing web search: {e}"

    def save_solution(self, filename: str, content: str) -> str:
        """
        Save content to a file.
        """
        try:
            safe_filename = os.path.basename(filename)
            file_path = os.path.join(config.OUTPUT_DIR, safe_filename)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
                
            logger.info(f"File saved: {file_path}")
            return f"File saved successfully: {file_path}"
        except Exception as e:
            logger.error(f"File save error: {e}")
            return f"Error saving file: {e}"

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge",
            "description": "Search local knowledge base for technical details, documentation, and project specifics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for up-to-date information, libraries, or errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_solution",
            "description": "Save the final answer, code, or documentation to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Name of the file (e.g., solution.md)."},
                    "content": {"type": "string", "description": "The content to write to the file."}
                },
                "required": ["filename", "content"]
            }
        }
    }
]

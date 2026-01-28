import os
import chromadb
from sentence_transformers import CrossEncoder
from ddgs import DDGS
from src.config import config
from src.utils import setup_logger, get_ollama_embedding

logger = setup_logger("Tools")

class ToolSet:
    def __init__(self, embedding_model: str = None):
        self.embedding_model = embedding_model if embedding_model else config.EMBEDDING_MODEL
        
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
        query_vector = get_ollama_embedding(query, model=self.embedding_model)
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
        Save the generated solution (code or guide) to a file.
        """
        file_path = os.path.join(config.OUTPUT_DIR, filename)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully saved to {file_path}"
        except Exception as e:
            return f"Error saving file: {e}"

    def create_plan(self, steps: list) -> str:
        """
        Create and save a step-by-step plan for the task.
        Use this tool to structure complex tasks before execution.
        """
        plan_content = "## Execution Plan\n\n"
        for i, step in enumerate(steps, 1):
            plan_content += f"{i}. {step}\n"
            
        logger.info(f"Plan Created: {steps}")
        return f"Plan created successfully with {len(steps)} steps:\n{plan_content}"

# Define Tools Schema for OpenAI/Ollama
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge",
            "description": "Search the local knowledge base for technical documentation and internal guidelines. ALWAYS use this first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for public information, library docs, or latest tech news. Use ONLY if local knowledge is insufficient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_solution",
            "description": "Save a generated code file, script, or markdown guide to the output directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file (e.g., 'script.py', 'guide.md')."
                    },
                    "content": {
                        "type": "string",
                        "description": "The text content to write to the file."
                    }
                },
                "required": ["filename", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_plan",
            "description": "Create a structured plan of action. REQUIRED for complex or multi-step tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of sequential steps to execute."
                    }
                },
                "required": ["steps"]
            }
        }
    }
]

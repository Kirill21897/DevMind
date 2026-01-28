import json
from openai import OpenAI
from langfuse.openai import OpenAI as LangfuseOpenAI
from langfuse import observe
from src.config import config
from src.utils import setup_logger
from .tools import ToolSet, TOOLS_SCHEMA
from .tracker import RagasTracker

logger = setup_logger("Agent")

class Agent:
    def __init__(self, system_prompt: str = None, model_name: str = None, embedding_model: str = None):
        self.model_name = model_name if model_name else config.LLM_MODEL
        self.embedding_model = embedding_model if embedding_model else config.EMBEDDING_MODEL
        logger.info(f"Initializing Agent with LLM: {self.model_name}, Embedding: {self.embedding_model}")
        
        # Check if LangFuse is configured
        if config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY:
            logger.info("LangFuse credentials found. Initializing LangFuse client.")
            self.client = LangfuseOpenAI(
                base_url=config.OLLAMA_BASE_URL,
                api_key="ollama"
            )
        else:
            logger.warning("LangFuse credentials NOT found. Using standard OpenAI client.")
            self.client = OpenAI(
                base_url=config.OLLAMA_BASE_URL,
                api_key="ollama"
            )
            
        self.tools = ToolSet(embedding_model=self.embedding_model)
        self.tracker = RagasTracker()
        
        default_prompt = """You are DevMind, an expert AI assistant strictly focused on software development and technical tasks.
        
        Your capabilities include:
        - Writing, debugging, and explaining code.
        - Designing software architecture and systems.
        - Searching technical documentation and solving engineering problems.
        - Researching latest technologies and tools.

        RULES:
        1. LANGUAGE:
           - If the user speaks Russian, YOU MUST REPLY IN RUSSIAN.
           - If the user speaks English, reply in English.
           - Maintain the user's language throughout the conversation.
        2. If a user asks about topics unrelated to software engineering, programming, or technology (e.g., politics, entertainment, cooking, general life advice), politely decline and state that you can only assist with technical tasks.
        3. PLANNING & REFLECTION (ReAct):
           - For complex tasks, use `create_plan` FIRST to outline your steps.
           - After each step, REFLECT: "Did I get what I needed? Do I need to change my plan?"
           - If a step fails, propose a fix or an alternative approach in your thought process.
           - If you are stuck or cannot find information, ask the user clarifying questions.
        4. TOOL USAGE:
           - Use `retrieve_knowledge` and `web_search` strategically to gather information needed for your plan.
           - Prefer local knowledge when appropriate, but you are free to choose the best tool for the current step.
        5. When asked to write code or guides, create high-quality markdown artifacts using the `save_solution` tool.
        6. Be concise, professional, and technically accurate.
        """
        
        self.system_prompt = system_prompt if system_prompt else default_prompt
        
        self.history = [{"role": "system", "content": self.system_prompt}]
        self.current_contexts = []

    @observe(name="Agent.run")
    async def run(self, user_query: str, callback=None) -> str:
        self.history.append({"role": "user", "content": user_query})
        self.current_contexts = []
        
        max_steps = 10
        step = 0
        
        while step < max_steps:
            step += 1
            try:
                # Synchronous call in async context (Ollama is fast enough locally)
                # Ideally, we should use AsyncOpenAI, but for now we keep it simple
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.history,
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    name="LLM-Generation" # LangFuse specific
                )
                
                message = response.choices[0].message
                
                if message.tool_calls:
                    self.history.append(message)
                    
                    for tool_call in message.tool_calls:
                        func_name = tool_call.function.name
                        args_str = tool_call.function.arguments
                        try:
                            args = json.loads(args_str)
                        except json.JSONDecodeError:
                            logger.error(f"JSON Decode Error for args: {args_str}")
                            args = {}

                        logger.info(f"Tool Call: {func_name}")
                        
                        # UI Callback for Tool Start
                        if callback:
                            await callback("tool_start", {"name": func_name, "args": args})

                        result = self._execute_tool(func_name, args)
                        
                        # UI Callback for Tool End
                        if callback:
                            await callback("tool_end", {"name": func_name, "result": result})

                        if func_name in ["retrieve_knowledge", "web_search"]:
                            self.current_contexts.append(str(result))
                        
                        self.history.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": func_name,
                            "content": str(result)
                        })
                else:
                    final_answer = message.content
                    self.history.append({"role": "assistant", "content": final_answer})
                    self.tracker.log_turn(user_query, final_answer, self.current_contexts)
                    return final_answer
                    
            except Exception as e:
                logger.error(f"Agent execution error: {e}")
                return f"Error during agent execution: {e}"
        
        return "Error: Maximum steps exceeded."

    @observe(as_type="generation")
    def _execute_tool(self, name: str, args: dict) -> str:
        try:
            if name == "retrieve_knowledge":
                return self.tools.retrieve_knowledge(args.get("query", ""))
            elif name == "web_search":
                return self.tools.web_search(args.get("query", ""))
            elif name == "save_solution":
                return self.tools.save_solution(args.get("filename", ""), args.get("content", ""))
            else:
                return f"Error: Unknown tool {name}"
        except Exception as e:
            return f"Error executing tool {name}: {e}"

import asyncio
import argparse
import sys
from src.agent import Agent
from src.utils import setup_logger

logger = setup_logger("Main")

async def run_chat(query: str = None):
    logger.info("Initializing DevMind Agent...")
    try:
        agent = Agent()
        print("\n" + "="*50)
        print("DevMind AI ready. Type 'exit' to quit.")
        print("="*50 + "\n")
        
        if query:
            print(f"User: {query}")
            print("Thinking...", end="", flush=True)
            ans = await agent.run(query)
            print(f"\rAgent: {ans}\n")
            return

        while True:
            try:
                q = input("User: ")
                if not q.strip():
                    continue
                if q.lower() in ["exit", "quit"]:
                    break
                
                print("Thinking...", end="", flush=True)
                ans = await agent.run(q)
                print(f"\rAgent: {ans}\n")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Runtime error: {e}")
                
    except Exception as e:
        logger.critical(f"Fatal Error during initialization: {e}")

def main():
    parser = argparse.ArgumentParser(description="DevMind AI Agent CLI")
    parser.add_argument("--query", type=str, help="Single query to run")
    args = parser.parse_args()

    try:
        asyncio.run(run_chat(args.query))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()

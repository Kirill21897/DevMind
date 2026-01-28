import chainlit as cl
from src.agent import Agent
from src.config import config
import logging

# Suppress debug logs
logging.basicConfig(level=logging.INFO)

@cl.on_chat_start
async def start():
    """
    Initializes the agent session.
    """
    agent = Agent()
    cl.user_session.set("agent", agent)
    
    await cl.Message(
        content=f"Hello! I am **DevMind**, your AI assistant powered by `{config.LLM_MODEL}`.\n\nI can help you with:\n- Searching your local knowledge base\n- Finding information on the web\n- Generating code and guides\n\nHow can I help you today?",
        author="DevMind"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """
    Handles incoming user messages.
    """
    agent = cl.user_session.get("agent")
    
    # Callback function to handle UI updates from the agent
    async def agent_callback(event_type, data):
        if event_type == "tool_start":
            tool_name = data["name"]
            # Create a step to show tool execution
            step = cl.Step(name=tool_name, type="tool")
            step.input = str(data["args"])
            await step.send()
            # Store step in session to update it later (optional, simplified here)
            cl.user_session.set(f"step_{tool_name}", step)
            
        elif event_type == "tool_end":
            tool_name = data["name"]
            result = data["result"]
            # Retrieve the step
            step = cl.user_session.get(f"step_{tool_name}")
            if step:
                step.output = str(result)
                await step.update()
    
    # Run agent asynchronously
    response = await agent.run(message.content, callback=agent_callback)
    
    # Send final answer
    await cl.Message(
        content=response,
        author="DevMind"
    ).send()

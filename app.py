import streamlit as st
import asyncio
import nest_asyncio
import os
import shutil
import subprocess
import sys
from src.agent import Agent
from src.config import config

# Apply nest_asyncio to allow nested event loops if necessary
nest_asyncio.apply()

# Page Config
st.set_page_config(
    page_title="DevMind AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "model_name" not in st.session_state:
    st.session_state.model_name = config.LLM_MODEL

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = config.EMBEDDING_MODEL

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hello! I am **DevMind**, your AI assistant powered by `{st.session_state.model_name}`.\n\nI can help you with:\n- Searching your local knowledge base\n- Finding information on the web\n- Generating code and guides\n\nHow can I help you today?"}
    ]

if "system_prompt" not in st.session_state:
    # Initialize with default prompt from Agent
    temp_agent = Agent()
    st.session_state.system_prompt = temp_agent.system_prompt

if "agent" not in st.session_state:
    st.session_state.agent = Agent(
        system_prompt=st.session_state.system_prompt, 
        model_name=st.session_state.model_name,
        embedding_model=st.session_state.embedding_model
    )

# Sidebar
with st.sidebar:
    st.title("🤖 DevMind AI")
    st.markdown("Your AI Pair Programmer")
    st.divider()
    
    # --- Knowledge Base Section ---
    with st.expander("📚 Knowledge Base", expanded=False):
        st.markdown(f"**Path:** `{config.DOCS_SOURCE_PATH}`")
        
        # File Uploader
        uploaded_files = st.file_uploader(
            "Upload Documents", 
            type=["md", "txt", "pdf"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("💾 Save Files"):
                if not os.path.exists(config.DOCS_SOURCE_PATH):
                    os.makedirs(config.DOCS_SOURCE_PATH)
                
                saved_count = 0
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(config.DOCS_SOURCE_PATH, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_count += 1
                st.success(f"Saved {saved_count} files!")

        st.divider()
        
        # Ingest Button
        if st.button("🔄 Re-index Knowledge Base"):
            status_container = st.status("Indexing documents...", expanded=True)
            try:
                # Run ingest script as subprocess
                script_path = os.path.join("scripts", "ingest_data.py")
                
                # Pass current embedding model via env var
                env = os.environ.copy()
                env["EMBEDDING_MODEL"] = st.session_state.embedding_model
                
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    cwd=os.getcwd(),
                    env=env
                )
                
                if result.returncode == 0:
                    status_container.update(label="Indexing Complete!", state="complete", expanded=False)
                    st.success(f"Knowledge base updated using {st.session_state.embedding_model}!")
                    with st.expander("See Output"):
                        st.code(result.stdout)
                else:
                    status_container.update(label="Indexing Failed", state="error", expanded=True)
                    st.error("Error during indexing")
                    st.code(result.stderr + "\n" + result.stdout)
                    
            except Exception as e:
                status_container.update(label="Error", state="error")
                st.error(f"Failed to run script: {e}")

    # --- Settings Section ---
    with st.expander("⚙️ Settings", expanded=False):
        # Model Selection
        st.subheader("Model Configuration")
        model_name = st.text_input("Ollama Model", value=st.session_state.model_name, help="Enter the Ollama model tag (e.g., qwen2.5-coder:14b)")
        embedding_model = st.text_input("Embedding Model", value=st.session_state.embedding_model, help="Enter the Ollama embedding model (e.g., nomic-embed-text)")
        
        st.divider()
        
        # System Prompt Editor
        st.subheader("System Prompt")
        new_prompt = st.text_area(
            "Edit System Prompt", 
            value=st.session_state.system_prompt,
            height=300
        )
        
        if st.button("Update Agent Configuration"):
            st.session_state.system_prompt = new_prompt
            st.session_state.model_name = model_name
            st.session_state.embedding_model = embedding_model
            
            st.session_state.agent = Agent(
                system_prompt=new_prompt, 
                model_name=model_name,
                embedding_model=embedding_model
            )
            
            # Reset history but keep welcome message
            welcome_msg = f"Hello! I am **DevMind**, your AI assistant powered by `{model_name}`.\n\nI can help you with:\n- Searching your local knowledge base\n- Finding information on the web\n- Generating code and guides\n\nHow can I help you today?"
            st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
            
            st.success(f"Agent updated! LLM: {model_name}, Embedding: {embedding_model}")
            st.rerun()

    st.divider()
    
    # Clear History Button
    if st.button("🗑️ Clear Chat History"):
        current_model = st.session_state.model_name
        welcome_msg = f"Hello! I am **DevMind**, your AI assistant powered by `{current_model}`.\n\nI can help you with:\n- Searching your local knowledge base\n- Finding information on the web\n- Generating code and guides\n\nHow can I help you today?"
        
        st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
        # Re-initialize agent to clear short-term memory
        st.session_state.agent = Agent(
            system_prompt=st.session_state.system_prompt, 
            model_name=current_model,
            embedding_model=st.session_state.embedding_model
        )
        st.rerun()

# --- Main Chat Interface ---

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent Execution
    with st.chat_message("assistant"):
        # Status container for tools
        status_container = st.status("Thinking...", expanded=True)
        
        async def streamlit_callback(event_type, data):
            if event_type == "tool_start":
                tool_name = data["name"]
                status_container.write(f"🔧 **Calling Tool:** `{tool_name}`")
            elif event_type == "tool_end":
                tool_name = data["name"]
                result = data["result"]
                status_container.write(f"✅ **Tool Result:** `{tool_name}`")
                with status_container.expander(f"See result for {tool_name}"):
                    st.code(result)

        try:
            # Run async agent
            response = asyncio.run(st.session_state.agent.run(prompt, callback=streamlit_callback))
            
            status_container.update(label="Response Ready!", state="complete", expanded=False)
            st.markdown(response)
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            status_container.update(label="Error Occurred", state="error", expanded=True)
            st.error(f"An error occurred: {str(e)}")

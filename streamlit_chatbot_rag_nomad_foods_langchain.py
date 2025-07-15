# streamlit_agent_app.py
import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import time

# Import the agent function
try:
    from rag_nomad_foods_chatbot_langchain import run_agent
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    IMPORT_ERROR = str(e)

# ─── 0. Page config: MUST be first Streamlit call ──────────────────
st.set_page_config(
    page_title="Nomad Foods AI Agent", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# ─── 1. Helper functions ────────────────────────────────────────────
def load_logo():
    """Load the Nomad Foods logo if it exists."""
    try:
        return Image.open("nomad_foods_logo.jpg")
    except FileNotFoundError:
        return None

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_ready" not in st.session_state:
        st.session_state.agent_ready = AGENT_AVAILABLE

def display_chat_message(role, content, avatar=None):
    """Display a chat message with proper formatting."""
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

def check_environment():
    """Check if all required environment variables are set."""
    required_vars = ["OPENROUTER_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX"]
    optional_vars = ["PINECONE_ENV"]  # Optional, defaults to us-east-1
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    return missing_vars, optional_vars

# ─── 2. Sidebar with information ───────────────────────────────────
def render_sidebar():
    """Render the sidebar with app information and controls."""
    with st.sidebar:
        st.header("🤖 Nomad Foods AI Agent")
        
        # Logo
        logo = load_logo()
        if logo:
            st.image(logo, width=200)
        
        st.markdown("---")
        
        # Environment check
        missing_vars, optional_vars = check_environment()
        if missing_vars:
            st.error(f"Missing required variables: {', '.join(missing_vars)}")
            st.markdown("Please check your `.env` file.")
        else:
            st.success("✅ Environment variables loaded")
            
        # Show optional variables status
        if not os.getenv("PINECONE_ENV"):
            st.info("ℹ️ Using default Pinecone region (us-east-1)")
        else:
            st.success(f"✅ Pinecone region: {os.getenv('PINECONE_ENV')}")
        
        # Agent status
        if AGENT_AVAILABLE:
            st.success("✅ Agent ready")
        else:
            st.error("❌ Agent unavailable")
            with st.expander("Error details"):
                st.code(IMPORT_ERROR)
        
        st.markdown("---")
        
        # Instructions
        st.markdown("""
        ### How to use:
        1. Type your question in the chat input below
        2. The agent will search the FAQ database first
        3. If no relevant info is found, it will search the web
        4. Get comprehensive answers about Nomad Foods
        
        ### Example questions:
        - What is Nomad Foods?
        - What brands does Nomad Foods own?
        - Where is Nomad Foods headquartered?
        - What are Nomad Foods' financial results?
        """)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

# ─── 3. Main chat interface ────────────────────────────────────────
def render_chat_interface():
    """Render the main chat interface."""
    st.title("🤖 Nomad Foods AI Agent")
    st.markdown("Ask me anything about Nomad Foods!")
    
    # Display chat messages
    for message in st.session_state.messages:
        display_chat_message(
            message["role"], 
            message["content"],
            avatar="🤖" if message["role"] == "assistant" else "👤"
        )
    
    # Chat input
    if prompt := st.chat_input("Type your question here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt, avatar="👤")
        
        # Generate assistant response
        with st.chat_message("assistant", avatar="🤖"):
            if not AGENT_AVAILABLE:
                response = "Sorry, the agent is currently unavailable. Please check the sidebar for error details."
                st.markdown(response)
            else:
                # Show thinking indicator
                with st.spinner("🤔 Thinking..."):
                    try:
                        # Add a small delay to show the spinner
                        time.sleep(0.5)
                        response = run_agent(prompt)
                        
                        # Handle empty or None responses
                        if not response or response.strip() == "":
                            response = "I'm sorry, I couldn't generate a response. Please try rephrasing your question."
                        
                    except Exception as e:
                        response = f"I apologize, but I encountered an error: {str(e)}. Please try again."
                
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# ─── 4. Error handling and fallbacks ──────────────────────────────
def render_error_page():
    """Render an error page when the agent is not available."""
    st.error("🚨 Agent Unavailable")
    st.markdown("""
    The Nomad Foods AI Agent is currently unavailable. This could be due to:
    
    - Missing dependencies
    - Configuration issues
    - Network connectivity problems
    
    Please check the sidebar for more details and try again later.
    """)

# ─── 5. Main application ───────────────────────────────────────────
def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render main interface
    if st.session_state.agent_ready:
        render_chat_interface()
    else:
        render_error_page()

# ─── 6. Run the application ────────────────────────────────────────
if __name__ == "__main__":
    main()
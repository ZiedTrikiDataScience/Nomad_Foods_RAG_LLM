import streamlit as st
from rag_nomad_foods_chatbot import search_similar_question, generate_enhanced_answer
import os
from PIL import Image
import time

#  Function to handle the chatbot logic
def chatbot(prompt):
    api_key = os.environ.get("MISTRAL_API_KEY")
    with st.spinner("🔍 Searching for relevant information..."):
        faq = search_similar_question(prompt)
        time.sleep(0.4)  # Add a small delay for UX
    with st.spinner("🤔 Generating enhanced response..."):
        enhanced_answer = generate_enhanced_answer(prompt, faq['answer'], api_key=api_key)
        time.sleep(0.4)  # Add a small delay for the UX
    return enhanced_answer


def load_assets():
    try:
        logo = Image.open('nomad_foods_logo.jpg')  # Use Nomad Foods logo
        return logo
    except FileNotFoundError:
        st.warning("Logo file not found. Please ensure 'nomad_foods_logo.jpg' exists in the correct path.")
        return None

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""  # Track the user query

def app():
    # Page configuration
    st.set_page_config(
        page_title="Nomad Foods AI Assistant",
        page_icon="🍽️",
        layout="wide"
    )

    initialize_session_state()
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stTextInput > div > div > input {
            font-size: 18px;
        }
        .st-emotion-cache-1helkxk p {
            font-size: 16px;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .user-message {
            background-color: #e6f2ff;
        }
        .bot-message {
            background-color: #f0f2f6;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create two columns: sidebar and main content
    col1, col2 = st.columns([1, 3])

    with col1:
        st.sidebar.header("⚙️ Settings")
        temperature = st.sidebar.slider("Response Creativity", 0.0, 1.0, 0.7)
        max_tokens = st.sidebar.slider("Response Length", 50, 500, 150)
        
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Statistics")
        st.sidebar.metric("Questions Asked", len(st.session_state.history))
        
        st.sidebar.markdown("---")
        st.sidebar.header("ℹ️ About")
        st.sidebar.info("""
        This AI assistant uses Retrieval-Augmented Generation (RAG) powered by MISTRAL AI
        to answer your queries related to Nomad Foods, covering various products and services.
        """)

    with col2:
        # Main content
        logo = load_assets()
        if logo:
            st.image(logo, width=350)
        
        st.title("Nomad Foods AI Assistant 🍽️")
        st.markdown("""
            <div style='background-color: #f0f7fb; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
                Welcome! I'm here to assist you with any questions about Nomad Foods' products 
                and services. Feel free to ask about anything from product information to delivery options.
            </div>
        """, unsafe_allow_html=True)

        # Sample questions for quick queries
        col1, col2, col3 = st.columns([1, 1, 1])
        sample_questions = [
            "What are the ingredients in Nomad Foods' products?",
            "Does Nomad Foods offer vegan options?",
            "How can I track my Nomad Foods order?"
        ]
        
        # Check if a sample question button is clicked
        for i, col in enumerate([col1, col2, col3]):
            if i < len(sample_questions):
                if col.button(f"📝 {sample_questions[i]}", key=f"sample_{i}"):
                    st.session_state.user_query = sample_questions[i]

        # Update text input value based on the selected sample question
        user_query = st.text_input("", placeholder="Type your question here please...", value=st.session_state.user_query, key="user_input")

        if user_query:
            response = chatbot(user_query)
            st.session_state.history.append({"query": user_query, "response": response})

        # Display conversation history
        if st.session_state.history:
            st.markdown("### 💬 Conversation History")
            for i, interaction in enumerate(reversed(st.session_state.history), 1):
                # User message
                st.markdown(f"""
                    <div class='chat-message user-message'>
                        <span style='font-weight: bold;'>You:</span>
                        {interaction['query']}
                    </div>
                """, unsafe_allow_html=True)
                
                # Bot message
                st.markdown(f"""
                    <div class='chat-message bot-message'>
                        <span style='font-weight: bold;'>AI Assistant:</span>
                        {interaction['response']}
                    </div>
                """, unsafe_allow_html=True)
                
                # Feedback buttons
                col1, col2 = st.columns([1, 5])
                with col1:
                    if f"thumbs_up_{i}" not in st.session_state.feedback:
                        if st.button("👍", key=f"up_{i}"):
                            st.session_state.feedback[f"thumbs_up_{i}"] = True
                            st.success("Thanks for your feedback!")
                    
                    if f"thumbs_down_{i}" not in st.session_state.feedback:
                        if st.button("👎", key=f"down_{i}"):
                            st.session_state.feedback[f"thumbs_down_{i}"] = True
                            st.error("Sorry for the inconvenience. We'll improve!")

        # Footer
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; color: #666666;'>
                Project Done by : ZIED TRIKI , 
                                    GenAI MLOps Engineer | 
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    app()

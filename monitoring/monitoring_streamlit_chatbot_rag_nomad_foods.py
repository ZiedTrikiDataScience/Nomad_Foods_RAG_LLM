import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import mistralai
from mistralai import Mistral
import os
import psycopg2
import streamlit as st
from PIL import Image
import time

# Function to insert feedback into PostgreSQL
def insert_feedback(user_query, thumbs_up, thumbs_down, relevant, model_used, response_time):
    conn = psycopg2.connect(
        dbname='monitoring_db',
        user='postgres',
        password='example',
        host='localhost',  
        port='5432'
    )
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback (user_query, thumbs_up, thumbs_down, relevant, model_used, response_time)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (user_query, thumbs_up, thumbs_down, relevant, model_used, response_time))
    conn.commit()
    cursor.close()
    conn.close()

# 1. Read Data from JSON File
with open('../faq_data.json', 'r') as file:
    qa_data = json.load(file)

# 2. Load the embedding model
model = SentenceTransformer("all-mpnet-base-v2")

# 3. Create FAISS index
embedding_dim = 768  # Dimensionality of embeddings from the model
faiss_index = faiss.IndexFlatL2(embedding_dim)  # L2 distance for similarity search

# Store questions and embeddings
questions = []
embeddings = []

# 4. Generate embeddings and add them to the FAISS index
for category in qa_data['faq_data']:  # Iterate over categories
    for qa in category['questions']:  # Iterate over questions in each category
        question = qa['question']
        embedding = model.encode(question)  # Generate vector embedding
        
        questions.append({"question": question, "answer": qa['answer']})
        embeddings.append(embedding)

# Convert embeddings to numpy array and add to FAISS index
embedding_matrix = np.array(embeddings)
faiss_index.add(embedding_matrix)

# 5. Function to search for the most similar question using FAISS
def search_similar_question(prompt):
    query_vector = model.encode(prompt)  # Convert user prompt to vector
    query_vector = np.array([query_vector]).astype('float32')

    # Search in FAISS
    _, indices = faiss_index.search(query_vector, 2)  # Search for 2 closest matches
    result_index = indices[0][0]  # Get the index of the best match
    
    # Return the question and answer corresponding to the matched index
    return questions[result_index]

# 6. Enhance response generation with MISTRAL AI
api_key = os.getenv('MISTRAL_API_KEY')

def generate_enhanced_answer(prompt, context, api_key):
    client = Mistral(api_key=api_key)
    
    # Define a more conversational system prompt
    system_prompt = """You are a friendly and helpful customer service representative at NomadFoods company. 
    Your responses should be warm, natural, and conversational while being informative.
    Use the following context to answer the question, but respond in a natural way as if you're 
    having a conversation. Avoid formal phrases like 'Based on the description provided' or 
    'Here's a summarized list'. Instead, use more conversational language like 'We at NomadFoods offer' or 
    'At NomadFoods, You can find'.

    Context: {context}
    """
    
    # Create a more natural user prompt
    enhanced_user_prompt = f"""Answer this question in a friendly, conversational way: {prompt}
    Make sure to:
    1. Start with a small warm greeting or acknowledgment
    2. Use natural transitions and conversational language
    3. Organize information in an easy-to-understand way with bullet points for example
    4. End with an offer to help further if needed
    """
    
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": system_prompt.format(context=context)},
            {"role": "user", "content": enhanced_user_prompt}
        ],
        max_tokens=500,
        temperature=0.7  # Slightly increased temperature for more natural responses
    )
    
    return response.choices[0].message.content.strip()

# Function to handle the chatbot logic
def chatbot(prompt):
    api_key = os.environ.get("MISTRAL_API_KEY")
    with st.spinner("🔍 Searching for relevant information..."):
        faq = search_similar_question(prompt)
        time.sleep(0.5)  # Add a small delay for UX
    with st.spinner("🤔 Generating enhanced response..."):
        enhanced_answer = generate_enhanced_answer(prompt, faq['answer'], api_key=api_key)
        time.sleep(0.5)  # Add a small delay for UX
    return enhanced_answer

def load_assets():
    try:
        logo = Image.open('../nomad_foods_logo.jpg')  # Use Nomad Foods logo
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
            response_time_start = time.time()
            response = chatbot(user_query)
            response_time = time.time() - response_time_start
            
            st.session_state.history.append({"query": user_query, "response": response})

        # Display conversation history
        if st.session_state.history:
            st.markdown("### 💬 Conversation History")
            for i, interaction in enumerate(reversed(st.session_state.history)):
                st.markdown(f"<div class='chat-message user-message'>{interaction['query']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-message bot-message'>{interaction['response']}</div>", unsafe_allow_html=True)

                # Feedback buttons
                if st.button("👍", key=f"up_{i}"):
                    insert_feedback(interaction['query'], True, False, True, "Mistral", response_time)
                    st.success("Thanks for your feedback!")

                if st.button("👎", key=f"down_{i}"):
                    insert_feedback(interaction['query'], False, True, False, "Mistral", response_time)
                    st.error("Sorry for the inconvenience. We'll improve!")

if __name__ == "__main__":
    app()

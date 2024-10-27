import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import mistralai
from mistralai import Mistral
import os

# 1. Read Data from JSON File
with open('faq_data.json', 'r') as file:
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
    _, indices = faiss_index.search(query_vector, 2)  # Search for 2 closest matchs
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


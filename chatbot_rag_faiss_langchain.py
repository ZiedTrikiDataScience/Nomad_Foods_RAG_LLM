import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from mistralai import Mistral
import faiss

# ------------------ RAG Components with FAISS & Mistral Integration ------------------

# Load FAQ Data
with open('faq_data.json', 'r') as file:
    qa_data = json.load(file)

# Load Sentence Transformer Model
model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name)

# Embed Questions for Similarity Search
questions = [qa["question"] for qa in qa_data]
embeddings = model.encode(questions)

# Initialize FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexHNSWFlat(dimension, 32)
index.add(embeddings)

# Function to search similar questions using FAISS
def search_similar_question(prompt):
    """
    Search for the most similar question in the FAQ using FAISS index.
    """
    # Convert prompt to embedding
    query_embedding = model.encode([prompt])
    # Search for the top 1 most similar questions
    distances, indices = index.search(query_embedding, k=1)
    matched_question = questions[indices[0][0]]
    matched_answer = qa_data[indices[0][0]]["answer"]
    return {"question": matched_question, "answer": matched_answer, "distance": distances[0][0]}

# Function to generate enhanced answer with Mistral AI
def generate_enhanced_answer(prompt, context, api_key, temperature=0.7, max_tokens=150):
    """
    Generate a context-enhanced answer using Mistral AI.
    """
    # Initialize Mistral Client
    mistral_client = Mistral(api_key=api_key)
    response = mistral_client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": f"Context: {context}"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

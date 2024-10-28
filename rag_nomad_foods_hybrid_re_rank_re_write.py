import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rag_nomad_foods_chatbot import search_similar_question, generate_enhanced_answer
import mistralai
from mistralai import Mistral
import os

# 1. Read Data from JSON File with error handling
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
if 'faq_data' in qa_data:
    for category in qa_data['faq_data']:
        if 'questions' in category:
            for qa in category['questions']:
                question = qa.get('question', '')
                answer = qa.get('answer', '')

                if question:  # Ensure question exists
                    embedding = model.encode(question)  # Generate vector embedding
                    questions.append({"question": question, "answer": answer})
                    embeddings.append(embedding)

# Convert embeddings to numpy array and add to FAISS index
embedding_matrix = np.array(embeddings, dtype="float32")
faiss_index.add(embedding_matrix)

# 5. Function to search for the most similar question using FAISS (basic method without enhancements)
def basic_faiss_search(prompt):
    query_vector = model.encode(prompt)  # Convert user prompt to vector
    query_vector = np.array([query_vector]).astype('float32')

    # Search in FAISS
    _, indices = faiss_index.search(query_vector, 1)  # Get top 1 closest match
    result_index = indices[0][0]  # Get the index of the best match
    
    # Return the question and answer corresponding to the matched index
    return questions[result_index]

# 6. Enhanced response generation with Mistral AI
api_key = os.getenv('MISTRAL_API_KEY')

def generate_enhanced_answer(prompt, context, api_key):
    client = Mistral(api_key=api_key)
    
    # System prompt for conversational response
    system_prompt = """You are a friendly and helpful customer service representative at NomadFoods company. 
    Your responses should be warm, natural, and conversational while being informative.
    Use the following context to answer the question.

    Context: {context}
    """
    
    # Create a natural, conversational prompt for the user query
    enhanced_user_prompt = f"""Answer this question in a friendly, conversational way: {prompt}
    Make sure to:
    1. Start with a small warm greeting or acknowledgment.
    2. Use natural transitions and conversational language.
    3. Organize information in an easy-to-understand way.
    4. End with an offer to help further if needed.
    """
    
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": system_prompt.format(context=context)},
            {"role": "user", "content": enhanced_user_prompt}
        ],
        max_tokens=500,
        temperature=0.7  
    )
    
    return response.choices[0].message.content.strip()

# 7. Re-ranking Function with Mistral AI
def re_rank_results(query, results, api_key):
    client = Mistral(api_key=api_key)
    
    ranking_prompt = f"""Rank the following answers based on relevance to this query: '{query}'. 
    Provide a sorted list, with the most relevant answer first:
    {results}"""
    
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": ranking_prompt}],
        max_tokens=150,
        temperature=0.5
    )
    
    # Check if re-ranking output is in the expected list format
    re_ranked_output = response.choices[0].message.content.strip()
    return re_ranked_output if isinstance(re_ranked_output, list) else []

# 8. Hybrid search with re-ranking
def hybrid_search(prompt):
    # Step 1: Perform FAISS search for initial similarity match
    initial_match = search_similar_question(prompt)
    initial_context = initial_match['answer']  
    
    # Step 2: Generate enhanced answer with Mistral AI
    enhanced_answer = generate_enhanced_answer(prompt, initial_context, api_key)
    
    # Step 3: Perform re-ranking based on multiple results
    re_ranked_results = re_rank_results(prompt, [initial_match], api_key)
    if re_ranked_results:
        return re_ranked_results[0]["answer"]
    else:
        return enhanced_answer

# 9. Comparison Function
def compare_hybrid_and_basic_faiss_vector_search(prompt):
    # Get basic FAISS result
    basic_result = basic_faiss_search(prompt)
    
    # Get hybrid search result
    hybrid_result = hybrid_search(prompt)
    
    # Print both results for comparison
    print("Basic FAISS Search Result:", basic_result['answer'] , "\n")
    print("Hybrid Search Result:", hybrid_result)

# Usage Evaluation
user_query = "How does NomadFoods handle refunds?"
compare_hybrid_and_basic_faiss_vector_search(user_query)

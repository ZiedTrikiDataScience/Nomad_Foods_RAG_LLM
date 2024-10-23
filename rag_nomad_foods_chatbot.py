import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from together import Together
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
    _, indices = faiss_index.search(query_vector, 2)  # Search for 1 closest match
    result_index = indices[0][0]  # Get the index of the best match
    
    # Return the question and answer corresponding to the matched index
    return questions[result_index]

# 6. Enhance response with Together AI
api_key = os.getenv('TOGETHER_API_KEY')

def generate_enhanced_answer(prompt, context , api_key):
    client = Together(api_key=api_key)
    response = client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
         messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
        ]
        
    )
    
    return response.choices[0].message.content.strip()

# Example usage:

#user_prompt = "i want to know the delivery times and the payment methods that you accept"
# 7. Retrieve similar question and answer from FAISS
#similar_qa = search_similar_question(user_prompt)
#context = similar_qa['answer']
# 8. Generate enhanced response with Together AI
#enhanced_answer = generate_enhanced_answer(user_prompt, context)
#print("Enhanced Answer:", enhanced_answer)

import json
import numpy as np
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import mistralai
from mistralai import Mistral
import os

# 0. Connect to ElasticSearch :
"""
es_username = "elastic"
es_password = os.getenv('es_password')

# Create an Elasticsearch client with authentication
es = Elasticsearch(
    ["http://localhost:9200"],
    basic_auth=(es_username, es_password)
)
"""
# Get the Elasticsearch host from environment variable
es_host = os.getenv('ELASTICSEARCH_HOST', 'http://localhost:9200')
#es_host = Elasticsearch("http://localhost:9200")

# Create an Elasticsearch client
es = Elasticsearch([es_host])

# Test the connection
try:
    if es.ping():
        print('\n' , "Successefully Connected to Elasticsearch!" , '\n')
    else:
        print("Could not connect to Elasticsearch")
except Exception as e:
    print(f"An error occurred: {e}")



# 1. Read Data from JSON File
with open('faq_data.json', 'r') as file:
    qa_data = json.load(file)


# 2.1 Prepare the index Mapping for Elastic Search :

index_name = 'fintechx_faq_vector'

index_mapping = {
    "mappings": {
        "properties": {
            "question": {"type": "text"},
            "answer": {"type": "text"},
            "question_vector": {"type": "dense_vector", "dims": 768 , "index" : True, "similarity": "cosine"}  
        }
    }
}

2.2 # Create the index in Elasticsearch
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=index_mapping)
else:
    es.indices.delete(index=index_name, ignore_unavailable= True)
    es.indices.create(index=index_name, body=index_mapping)    



# 3.1 Load the embedding model
model = SentenceTransformer("all-mpnet-base-v2")

# 3.2. Create the embeddings with the loaded model : 
for i, qa in enumerate(qa_data):
    question = qa['question']
    embedding = model.encode(question)  # Generate the vector embedding

    # 3.3.  Create a document with the question, answer, and vector
    doc = {
        "question": question,
        "answer": qa['answer'],
        "question_vector": embedding.tolist() } # Convert question vectors to list
    
    # 3.4 . Index the created document in Elasticsearch
    es.index(index=index_name, id=i, body=doc)




# 4.1 Convert the user prompt into a vector and search for similar vector embeddings :

def search_similar_question(prompt):
    query_vector = model.encode(prompt)  # Convert user prompt to vector

    # 4.2: Define a query for vector search using cosine similarity
    query = {
        "size": 1,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'question_vector') + 1.0",
                    "params": {"query_vector": query_vector.tolist()}
                }
            }
        }
    }

    
    # 4.3: Perform the search in Elasticsearch
    response = es.search(index=index_name, body=query)
    return response['hits']['hits'][0]['_source']



# 8. Enhance response with MISTRAL AI : 

api_key = os.getenv('MISTRAL_API_KEY')


"""
def generate_enhanced_answer(prompt, context):
    response = mistralai.Completion.create(
        model="mistral",
        prompt=f"{context}\nUser: {prompt}\nBot:",
        max_tokens=150
    )
    return response['choices'][0]['text'].strip()
"""


def generate_enhanced_answer(prompt, context, api_key):
    client = Mistral(api_key=api_key)
    
    response = client.chat.complete(
        model="mistral-large-latest", 
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    
    return response.choices[0].message.content.strip()
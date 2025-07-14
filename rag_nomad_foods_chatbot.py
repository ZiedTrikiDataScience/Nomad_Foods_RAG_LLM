import os
import json
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec, CloudProvider, VectorType

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Don't decorate this with @st.cache_* here
def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx["name"] for idx in pc.list_indexes()]
    if PINECONE_INDEX not in existing:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud=CloudProvider.AWS, region=PINECONE_REGION),
            vector_type=VectorType.DENSE
        )
    return pc.Index(name=PINECONE_INDEX)

def load_model():
    return SentenceTransformer("all-mpnet-base-v2")

def upsert_faq(index, model):
    with open("faq_data.json", "r") as f:
        qa_data = json.load(f)

    stats = index.describe_index_stats()
    if stats["total_vector_count"] == 0:
        batch = []
        for i, cat in enumerate(qa_data["faq_data"]):
            for j, qa in enumerate(cat["questions"]):
                vec = model.encode(qa["question"]).tolist()
                batch.append({
                    "id": f"faq_{i}_{j}",
                    "values": vec,
                    "metadata": {"question": qa["question"], "answer": qa["answer"]}
                })
        index.upsert(vectors=batch)

# These are the ones you expose
def search_similar_question(prompt):
    index = init_pinecone()
    model = load_model()
    upsert_faq(index, model)
    q_vec = model.encode(prompt).tolist()
    resp = index.query(
        vector=q_vec,
        top_k=1,
        include_metadata=True
    )
    match = resp["matches"][0]
    return {"question": match["metadata"]["question"], "answer": match["metadata"]["answer"]}

def generate_enhanced_answer(prompt, context, api_key):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/ZiedTrikiDataScience/Nomad_Foods_RAG_LLM",
        "X-Title": "NomadFoods FAQ Assistant"
    }
    system_prompt = f"""
    You are a friendly and helpful customer service representative at NomadFoods company.
    Your responses should be warm, natural, and conversational while being informative.
    Context: {context}
    """
    enhanced_user_prompt = f"""
    Answer this question in a friendly, conversational way: {prompt}
    - Start with a warm greeting
    - Use easy-to-understand language
    - Use bullet points
    - Offer to help more at the end
    """
    payload = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": enhanced_user_prompt.strip()}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    r = requests.post(url, headers=headers, json=payload)
    if r.status_code != 200:
        return f"❌ OpenRouter error {r.status_code}: {r.text}"
    return r.json()["choices"][0]["message"]["content"].strip()

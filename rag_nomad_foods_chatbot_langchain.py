# rag_agent_functions.py
import os, json, requests
from dotenv import load_dotenv

# LangChain and LlamaIndex
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chat_models import ChatOpenAI
from llama_index.core import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
)
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from langchain_community.tools import DuckDuckGoSearchRun
from pinecone import Pinecone, ServerlessSpec, CloudProvider, VectorType

# ─── 0. Load credentials ─────────────────────────────────────────
load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_ENV   = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# ─── 1. Build LLM (function, but uses ChatOpenAI under the hood) ────
def build_llm():
    return ChatOpenAI(
        model_name="deepseek/deepseek-chat-v3-0324:free",
        temperature=0.7,
        openai_api_key=OPENROUTER_KEY,
        openai_api_base="https://openrouter.ai/api/v1"
    )

# ─── 2. Build Vector Store + LlamaIndex ────────────────────────────
def build_index():
    # init Pinecone
    pc = Pinecone(api_key=PINECONE_KEY)
    if PINECONE_INDEX not in [idx["name"] for idx in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud=CloudProvider.AWS, region=PINECONE_ENV),
            vector_type=VectorType.DENSE
        )
    pinecone_index = pc.Index(name=PINECONE_INDEX)

    # wrap for LlamaIndex
    vstore = PineconeVectorStore(pinecone_index, namespace="")
    svc_ctx = ServiceContext.from_defaults(embed_model=LangchainEmbedding())
    stor_ctx = StorageContext.from_defaults(vector_store=vstore)

    # ingest if empty
    if vstore.count() == 0:
        with open("faq_data.json") as f:
            docs = []
            for cat in json.load(f)["faq_data"]:
                for qa in cat["questions"]:
                    docs.append({
                        "doc_id": qa["question"],
                        "text": qa["question"],
                        "metadata": {"answer": qa["answer"]}
                    })
        VectorStoreIndex.from_documents(
            docs,
            service_context=svc_ctx,
            storage_context=stor_ctx
        )
    return stor_ctx, svc_ctx

# ─── 3. Retrieval Tool (FAQs) ───────────────────────────────────────
def retrieve_faq(question: str) -> str:
    stor_ctx, svc_ctx = build_index()
    idx = VectorStoreIndex.from_storage_context(stor_ctx, service_context=svc_ctx)
    resp = idx.as_query_engine().query(question)
    return str(resp)

# ─── 4. Web‑Search Tool ──────────────────────────────────────────────
ddg = DuckDuckGoSearchRun()
def web_search(question: str) -> str:
    return ddg.run(question)

# ─── 5. Assemble and return the Agent runner ───────────────────────
def get_agent():
    llm = build_llm()

    tools = [
        Tool(
            name="FAQ Retriever",
            func=retrieve_faq,
            description="Use for answering questions from the Nomad Foods FAQ data."
        ),
        Tool(
            name="Web Search",
            func=web_search,
            description="Use when the question is not covered in the FAQ; returns a short web snippet."
        ),
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False
    )
    return agent.run  # this is a function you can call

# ─── 6. Exported entrypoint ──────────────────────────────────────────
run_agent = get_agent()

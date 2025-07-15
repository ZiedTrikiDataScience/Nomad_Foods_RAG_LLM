import os
import json
from dotenv import load_dotenv

# LangChain core agents/tools
from langchain.agents import initialize_agent, AgentType, Tool

# OpenAI chat model (new package)
from langchain_openai import ChatOpenAI

# Try the new import first, fallback to old one if not available
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.tools import DuckDuckGoSearchRun

# LlamaIndex and Pinecone
from llama_index.core import VectorStoreIndex, ServiceContext, StorageContext, Settings, Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# ─── 0. Load credentials ───────────────────────────────────────── ─────────────────────────────────────────
load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_KEY   = os.getenv("PINECONE_API_KEY")
PINECONE_ENV   = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Global variables to cache expensive operations
_embed_model = None
_index_cache = None

# ─── 1. Build and register embedding ─────────────────────────────
def build_embed_model() -> LangchainEmbedding:
    global _embed_model
    if _embed_model is None:
        hf = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        _embed_model = LangchainEmbedding(hf)
        Settings.embed_model = _embed_model
    return _embed_model

# ─── 2. Build Vector Store + LlamaIndex ────────────────────────────
def build_index():
    global _index_cache
    if _index_cache is not None:
        return _index_cache
    
    try:
        embed_model = build_embed_model()
        
        # init Pinecone and ensure index exists
        pc = Pinecone(api_key=PINECONE_KEY)
        
        # Check if index exists
        existing_indexes = [idx["name"] for idx in pc.list_indexes()]
        if PINECONE_INDEX not in existing_indexes:
            # Use default region if PINECONE_ENV is not set
            region = PINECONE_ENV or "us-east-1"
            pc.create_index(
                name=PINECONE_INDEX,
                dimension=384,  # dimension for all-mpnet-base-v2
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=region)
            )
        
        pinecone_index = pc.Index(name=PINECONE_INDEX)
        
        # wrap for LlamaIndex storage context
        vstore = PineconeVectorStore(pinecone_index, namespace="")
        stor_ctx = StorageContext.from_defaults(vector_store=vstore)
        
        # Check if index has data
        stats = pinecone_index.describe_index_stats()
        vector_count = stats.get("namespaces", {}).get("", {}).get("vector_count", 0)
        
        if vector_count == 0:
            # Load and ingest data
            if os.path.exists("faq_data.json"):
                with open("faq_data.json") as f:
                    data = json.load(f)
                    documents = []
                    for category in data.get("faq_data", []):
                        for qa in category.get("questions", []):
                            doc = Document(
                                text=f"Q: {qa['question']}\nA: {qa['answer']}",
                                metadata={"question": qa["question"], "answer": qa["answer"]}
                            )
                            documents.append(doc)
                
                if documents:
                    VectorStoreIndex.from_documents(documents, storage_context=stor_ctx)
        
        _index_cache = stor_ctx
        return stor_ctx
    
    except Exception as e:
        print(f"Error building index: {e}")
        return None

# ─── 3. Retrieval Tool (FAQs) ───────────────────────────────────────
def retrieve_faq(question: str) -> str:
    try:
        stor_ctx = build_index()
        if stor_ctx is None:
            return "Sorry, I couldn't access the FAQ database at the moment."
        
        idx = VectorStoreIndex.from_storage_context(stor_ctx)
        query_engine = idx.as_query_engine(similarity_top_k=3)
        response = query_engine.query(question)
        
        return str(response)
    except Exception as e:
        return f"Error retrieving FAQ: {str(e)}"

# ─── 4. Web‑Search Tool ──────────────────────────────────────────────
def create_web_search_tool():
    try:
        ddg = DuckDuckGoSearchRun()
        def web_search(question: str) -> str:
            try:
                return ddg.run(question)
            except Exception as e:
                return f"Web search error: {str(e)}"
        
        return Tool(
            name="Web Search",
            func=web_search,
            description="Use when the question is not covered in the FAQ; returns web search results."
        )
    except ImportError:
        def web_search(_: str) -> str:
            return "⚠️ Web search unavailable (duckduckgo-search not installed)."
        
        return Tool(
            name="Web Search (disabled)",
            func=web_search,
            description="DuckDuckGo search dependency missing; returns placeholder text."
        )

# ─── 5. Build LLM ───────────────────────────────────────────────────
def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek/deepseek-chat-v3-0324:free",
        temperature=0.7,
        openai_api_key=OPENROUTER_KEY,
        openai_api_base="https://openrouter.ai/api/v1"
    )

# ─── 6. Assemble and return the Agent executor ─────────────────────
def get_agent():
    try:
        llm = build_llm()
        faq_tool = Tool(
            name="FAQ Retriever",
            func=retrieve_faq,
            description="Use for answering questions about Nomad Foods from the FAQ database."
        )
        
        search_tool = create_web_search_tool()
        
        executor = initialize_agent(
            tools=[faq_tool, search_tool],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=3,
            max_execution_time=30
        )
        return executor
    except Exception as e:
        print(f"Error creating agent: {e}")
        return None

# ─── 7. Main function to run the agent ─────────────────────────────────────
def run_agent(query: str) -> str:
    """
    Run the agent with a query and return a clean response.
    """
    try:
        agent = get_agent()
        if agent is None:
            return "Sorry, I couldn't initialize the agent. Please try again later."
        
        # Run the agent
        result = agent.invoke({"input": query})
        
        # Extract the output from the result
        if isinstance(result, dict):
            output = result.get("output", str(result))
        else:
            output = str(result)
        
        # Clean up the output
        if "Final Answer:" in output:
            output = output.split("Final Answer:")[-1].strip()
        
        return output
    
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."

# For testing
if __name__ == "__main__":
    test_query = "What is Nomad Foods?"
    response = run_agent(test_query)
    print(f"Query: {test_query}")
    print(f"Response: {response}")
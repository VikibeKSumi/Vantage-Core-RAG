import os
import sys
import time
import yaml
import streamlit as st
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ai_core import AICore
from src.search_logic import SemanticSearcher
from src.generation import Generator
from src.database import VectorDBManager
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



# Page Config
st.set_page_config(page_title="Vantage Core-RAG", page_icon="🏗️", layout="wide")

# Load Environment
load_dotenv()
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

# Title & Sidebar
st.title("🏗️ Vantage Core-RAG")
st.caption("Bi-Encoder Optimized Two-Stage Retrieval Pipeline")

with st.sidebar:
    st.header("System Status")
    st.success("Core Engine: Online")
    st.info(f"Model: {config['models']['embedding']}")
    st.info(f"Reranker: {config['models']['reranker']}")
    
    # Simple metric display
    st.divider()
    st.markdown("### ⚡ Performance")
    latency_placeholder = st.empty()
    latency_placeholder.metric("Last Query Latency", "0.0s")

# Initialize Engine (Cached to prevent reloading on every click)
@st.cache_resource
def init_engine():
    # 1. Initialize AI Core (The Muscle)
    # We must build this object first because SemanticSearcher needs it.
    ai_core = AICore(config)
    
    # 2. Initialize Searcher (The Eyes)
    # CORRECT FIX: Pass the 'ai_core' object, not 'config'
    searcher = SemanticSearcher(ai_core)
    
    # 3. Initialize Generator (The Voice)
    model_name = config['models']['llm']
    generator = Generator(api_key=os.getenv("GROQ_API_KEY"), model=model_name)
    
    # 4. Load the Database Index (The Memory)
    # We need to pass this index to the searcher later
    db_manager = VectorDBManager(config)
    vector_store = db_manager.get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # LlamaIndex needs its own embed model wrapper
    embed_model = HuggingFaceEmbedding(
        model_name=config['models']['embedding'], 
        device="cpu"
    )
    
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        storage_context=storage_context, 
        embed_model=embed_model
    )
    
    return searcher, generator, index

try:
    searcher, generator, index = init_engine()
except Exception as e:
    st.error(f"Engine Validation Failed: {e}")
    st.stop()

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask the Architect..."):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. AI Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Start Timer
        start_time = time.time()
        
        with st.spinner("Analyzing Architecture..."):
         
            # A. Search & Rerank (Optimized)
            # Only fetch 5 candidates (instead of 10) to save CPU cycles
            retrieved_nodes = searcher.search_and_rank(prompt, index, top_k=5)
            
            # B. Generate
            if not retrieved_nodes:
                full_response = "No relevant architectural data found in the vault."
            else:
                full_response = generator.generate_response(prompt, retrieved_nodes)
        
        # End Timer
        end_time = time.time()
        latency = round(end_time - start_time, 2)
        latency_placeholder.metric("Last Query Latency", f"{latency}s")

        # C. Display & Save
        message_placeholder.markdown(full_response)
        
        # D. Show Sources (The "Architect" Proof)
        with st.expander("View Strategic Context (Sources)"):
            for i, node in enumerate(retrieved_nodes):
                score = node.score if hasattr(node, 'score') else "N/A"
                st.markdown(f"**Source {i+1} (Relevance: {score:.4f}):**")
                st.caption(node.get_content()[:300] + "...")

        st.session_state.messages.append({"role": "assistant", "content": full_response})
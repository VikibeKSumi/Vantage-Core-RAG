import sys
from pathlib import Path
import time
import streamlit as st

# Fix import path when running with Streamlit
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
from src.engine_load import RAGEngine

st.set_page_config(page_title="Vantage Core-RAG", page_icon="🏗️", layout="wide")

st.title("🏗️ vantageCoreRAG")
st.caption("Bi-Encoder + Cross-Encoder • Indic-Optimized • GPU Ready")

# Initialize engine once (cached)
@st.cache_resource
def get_engine():
    return RAGEngine()

engine = get_engine()

# 1. Initialize Chat History State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. Handle User Input & Inference
if prompt := st.chat_input("Ask about Budget 2026-27 or Economic Survey..."):
    # Append and show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and show assistant response
    with st.chat_message("assistant"):
        start_time = time.time()
        
        with st.spinner("Thinking..."):
            metrics = engine.ask(prompt, verbose=False)
        
        latency = round(time.time() - start_time, 2)
        st.markdown(metrics["answer"])
        
        # Update metrics in session state *before* sidebar rendering
        st.session_state.last_metrics = metrics
        
        # Sources
        with st.expander("📚 View Retrieved Sources"):
            for i, node in enumerate(metrics.get("retrieved_nodes", []), 1):
                score = getattr(node, 'score', 'N/A')
                st.markdown(f"**Source {i}** (Score: {score if isinstance(score, str) else f'{score:.4f}'})")
                st.caption(node.node.get_content()[:400] + "...")

    # Append assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": metrics["answer"]})

# 4. Render Sidebar Last (Reads the freshest session_state)
with st.sidebar:
    st.header("System Status")
    st.success("✅ Engine Online")
    st.info(f"Device: {engine.device.upper()}")
    
    st.divider()
    st.markdown("### 📊 Last Query Metrics")
    
    if "last_metrics" in st.session_state:
        m = st.session_state.last_metrics
        st.metric("Total Latency", f"{m.get('total_latency', 0):.2f}s")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Retrieval", f"{m.get('retrieval_time', 0):.2f}s")
            st.metric("Top Rerank", f"{m.get('top_rerank_score', 0):.4f}")
        with col2:
            st.metric("Generation", f"{m.get('generation_time', 0):.2f}s")
            st.metric("Avg Rerank", f"{m.get('avg_rerank_score', 0):.4f}")
        
        st.metric("Tokens Used", m.get('total_tokens', 0))
        if 'peak_vram_mb' in m:
            st.metric("Peak VRAM", f"{m['peak_vram_mb']} MB")
    else:
        st.info("Ask a question to see live metrics")
# api.py — Production Observability API (lightweight + no Qdrant conflict)
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import uvicorn
import time

app = FastAPI(title="Vantage Core RAG - Observability API")

@app.get("/health")
async def health():
    """Production health check"""
    return {
        "status": "healthy",
        "service": "vantage_core_rag",
        "timestamp": time.time()
    }

@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics"""
    return PlainTextResponse(f"""# HELP vantage_rag_info Vantage Core RAG status
        # TYPE vantage_rag_info gauge
        vantage_rag_info{{status="running",version="1.0"}} 1

        # HELP vantage_rag_uptime_seconds Uptime of the API
        # TYPE vantage_rag_uptime_seconds gauge
        vantage_rag_uptime_seconds {int(time.time())}

        # HELP vantage_rag_note Note for local setup
        # TYPE vantage_rag_note gauge
        vantage_rag_note{{note="Full metrics available in Streamlit UI"}} 1
    """)

if __name__ == "__main__":
    print("🚀 Starting Vantage Core Observability API on http://127.0.0.1:8000")
    print("   Health  → http://127.0.0.1:8000/health")
    print("   Metrics → http://127.0.0.1:8000/metrics")
    uvicorn.run(app, host="0.0.0.0", port=8000)
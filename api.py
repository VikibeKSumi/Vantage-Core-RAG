

from fastapi import FastAPI
from src.engine import Engine
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    

app = FastAPI(title="Policy Analysis")
engine = Engine()


@app.post('/ask')
def ask(query: QueryRequest):
    response = engine.run(query.query)
    return response


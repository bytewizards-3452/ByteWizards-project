import os
import uvicorn
import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "healthcare_docs", "index.faiss")
DOC_MAP_PATH = os.path.join(BASE_DIR, "healthcare_docs", "doc_map.json")

# --- GLOBAL VARIABLES ---
app = FastAPI()
model = None
index = None
doc_map = None

class QueryRequest(BaseModel):
    question: str

# --- STARTUP EVENT (Loads Everything) ---
@app.on_event("startup")
def startup_event():
    global model, index, doc_map
    
    # Load FAISS index and document map
    if os.path.exists(INDEX_PATH) and os.path.exists(DOC_MAP_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(DOC_MAP_PATH, 'r') as f:
            doc_map = {int(k): v for k, v in json.load(f).items()}
        print("✅ FAISS index and document map loaded.")
    else:
        print("🚨 Warning: Index/doc_map not found.")

    # Load the embedding model at startup
    print("Loading sentence-transformer model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("✅ AI Model loaded and ready.")


# --- QUERY ENDPOINT ---
@app.post("/query")
def query(req: QueryRequest):
    if index is None or model is None:
        return {"error": "Server is not ready. Please check logs."}

    query_embedding = model.encode([req.question], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k=1)
    
    match_index = indices[0][0]
    source_file = doc_map.get(match_index, "Unknown Source")
    answer_text = f"Found a relevant section in the document: {source_file}"

    return {"answer": answer_text, "source": source_file, "score": float(distances[0][0])}

# --- ROOT ENDPOINT ---
@app.get("/")
def root():
    return FileResponse("index.html")

# --- MAIN RUNNER ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

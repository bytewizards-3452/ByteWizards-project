import os
import uvicorn
import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.faiss")
DOC_MAP_PATH = os.path.join(BASE_DIR, "doc_map.npy")

app = FastAPI()

model = None
index = None
doc_map = None

class QueryRequest(BaseModel):
    question: str

@app.on_event("startup")
def startup_event():
    global index, doc_map
    print("--- Server Startup ---")
    print(f"Base directory detected as: {BASE_DIR}")
    print(f"Attempting to load index from: {INDEX_PATH}")
    print(f"Attempting to load doc_map from: {DOC_MAP_PATH}")

    if os.path.exists(INDEX_PATH) and os.path.exists(DOC_MAP_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            doc_map_array = np.load(DOC_MAP_PATH, allow_pickle=True)
            doc_map = doc_map_array.item()
            print("✅ FAISS index and document map loaded successfully.")
            print(f"Index contains {index.ntotal} vectors.")
        except Exception as e:
            print(f"🚨 Error loading files: {e}")
    else:
        print("🚨 Warning: Could not find 'index.faiss' and/or 'doc_map.npy'.")
        if not os.path.exists(INDEX_PATH):
            print("-> 'index.faiss' is MISSING.")
        if not os.path.exists(DOC_MAP_PATH):
            print("-> 'doc_map.npy' is MISSING.")
        print("The app will run, but querying will fail.")
    print("--- Startup Complete ---")


@app.post("/query")
def query(req: QueryRequest):
    global model, index, doc_map

    if index is None or doc_map is None:
        return {"error": "FAISS index is not available. Please check server logs."}

    if model is None:
        print("First query received. Loading sentence-transformer model...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Model loaded.")

    query_embedding = model.encode([req.question], convert_to_numpy=True)

    distances, indices = index.search(query_embedding, k=1)
    
    match_index = indices[0][0]
    
    answer_text = "No relevant document found."
    source_file = "N/A"
    
    if match_index < len(doc_map):
        source_file = doc_map.get(match_index, "Unknown Source")
        answer_text = f"Found a relevant section in the document: {source_file}"

    return {
        "answer": answer_text,
        "source": source_file,
        "score": float(distances[0][0])
    }

@app.get("/")
def root():
    frontend_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"error": "index.html not found."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

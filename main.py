import os
import uvicorn
import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import json

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

# --- STARTUP EVENT ---
@app.on_event("startup")
def startup_event():
    global index, doc_map
    print("--- Server Startup ---")
    
    if os.path.exists(INDEX_PATH) and os.path.exists(DOC_MAP_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            with open(DOC_MAP_PATH, 'r') as f:
                doc_map = {int(k): v for k, v in json.load(f).items()}
            print("✅ FAISS index and document map loaded successfully.")
            print(f"Index contains {index.ntotal} vectors.")
        except Exception as e:
            print(f"🚨 Error loading files: {e}")
    else:
        print("🚨 Warning: Index and/or document map file not found.")

# --- QUERY ENDPOINT ---
@app.post("/query")
def query(req: QueryRequest):
    global model, index, doc_map

    if index is None:
        return {"error": "Index not loaded."}

    if model is None:
        print("Loading embedding model on first query...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Model loaded.")

    query_embedding = model.encode([req.question], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k=1)
    
    match_index = indices[0][0]
    source_file = doc_map.get(match_index, "Unknown Source")
    answer_text = f"Found a relevant section in the document: {source_file}"

    return {
        "answer": answer_text,
        "source": source_file,
        "score": float(distances[0][0])
    }

# --- ROOT ENDPOINT (with debugging) ---
@app.get("/")
def root():
    print("\n--- A user visited the website root ('/') ---")
    
    frontend_path = os.path.join(BASE_DIR, "index.html")
    print(f"Checking for frontend file at this exact path: {frontend_path}")
    
    file_exists = os.path.exists(frontend_path)
    print(f"Result of os.path.exists() check: {file_exists}")

    if file_exists:
        print("✅ Success! Serving index.html.")
        return FileResponse(frontend_path)
    else:
        print("🚨 CRITICAL: index.html not found at the path.")
        print("Listing all files and folders found in the base directory:")
        try:
            files_in_directory = os.listdir(BASE_DIR)
            print(files_in_directory)
            return JSONResponse(
                status_code=404, 
                content={
                    "error": "index.html was not found by the server.",
                    "checked_path": frontend_path,
                    "files_found_in_directory": files_in_directory
                }
            )
        except Exception as e:
            print(f"Error trying to list directory files: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "A server error occurred while trying to list files."}
            )

# --- MAIN RUNNER ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

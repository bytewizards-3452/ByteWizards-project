from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
print("CWD:", os.getcwd())
print("Files in CWD:", os.listdir("."))

app = FastAPI()

model = None
index = None
doc_map = {}  # maps FAISS IDs to document metadata if needed

@app.on_event("startup")
def startup_event():
    global index, doc_map
    # Load FAISS index once at startup
    if os.path.exists("data/healthcare_docs/index.faiss"):
        index = faiss.read_index("data/healthcare_docs/index.faiss")
        # If you have a mapping file for doc texts/IDs, load it here:
        if os.path.exists("data/healthcare_docs/doc_map.npy"):
            doc_map = np.load("data/healthcare_docs/doc_map.npy", allow_pickle=True).item()
    else:
        print("Warning: FAISS index not found")

@app.get("/query")
def query_docs(q: str):
    global model
    if model is None:
        # Load the embedding model lazily
        model = SentenceTransformer("all-MiniLM-L6-v2")
    if index is None:
        return {"error": "FAISS index not loaded"}
    
    # Encode query → search FAISS index
    embedding = model.encode([q])
    D, I = index.search(np.array(embedding, dtype=np.float32), k=3)

    results = []
    for idx, dist in zip(I[0], D[0]):
        doc_info = doc_map.get(idx, f"Document {idx}")
        results.append({"doc": doc_info, "score": float(dist)})
    
    return {"query": q, "results": results}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=port)

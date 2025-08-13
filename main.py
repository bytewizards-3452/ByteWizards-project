from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import faiss, os, uvicorn, json
import numpy as np

app = FastAPI()

# Mount static folder for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

DATA_DIR = "data/healthcare_docs"
INDEX_FILE = "data/healthcare_index.faiss"
DOCS_META = "data/healthcare_docs.json"
EMBED_SIZE = 384  # adjust depending on your embedding model

docs = []
index = None

class QueryRequest(BaseModel):
    question: str
    mode: str

@app.on_event("startup")
def load_index():
    global index, docs
    if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_META):
        index = faiss.read_index(INDEX_FILE)
        with open(DOCS_META, "r") as f:
            docs = json.load(f)
    else:
        index = faiss.IndexFlatL2(EMBED_SIZE)
        docs = []

@app.post("/query")
def query(req: QueryRequest):
    # Dummy vector for demonstration (replace with real embeddings)
    vec = np.random.rand(1, EMBED_SIZE).astype("float32")
    if index.ntotal == 0:
        return {"answer": "No documents indexed yet."}
    D, I = index.search(vec, k=1)
    return {"answer": docs[I[0][0]] if I[0][0] < len(docs) else "No match found."}

@app.get("/")
def root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import faiss, os, uvicorn, json
import numpy as np

from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for PDFs
import docx  # python-docx for Word files

# ---- Config ----
DATA_DIR = "data/healthcare_docs"
INDEX_FILE = "data/healthcare_index.faiss"
DOCS_META = "data/healthcare_docs.json"
EMBED_SIZE = 384  # embedding dim for all-MiniLM-L6-v2

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---- Load embedding model ----
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---- Data ----
docs = []        # Stores extracted text snippets
doc_names = []   # Stores file names
index = None

class QueryRequest(BaseModel):
    question: str
    mode: str

# ---- Utility to extract text from files ----
def extract_text_from_file(path):
    if path.lower().endswith(".pdf"):
        text = ""
        with fitz.open(path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text.strip()
    elif path.lower().endswith(".docx"):
        d = docx.Document(path)
        return "\n".join([p.text for p in d.paragraphs if p.text.strip()])
    elif path.lower().endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return ""

# ---- Build FAISS index from scratch ----
def build_index():
    global index, docs, doc_names
    docs, doc_names = [], []
    texts = []
    for fname in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.isfile(fpath) and fname.lower().endswith((".pdf",".docx",".txt")):
            content = extract_text_from_file(fpath)
            if content:
                docs.append(content[:1000])  # store only snippet (first 1k chars)
                doc_names.append(fname)
                texts.append(content)
    if not texts:
        index = faiss.IndexFlatL2(EMBED_SIZE)
        return

    vectors = embedder.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(EMBED_SIZE)
    index.add(vectors)

    # Save FAISS index and docs
    faiss.write_index(index, INDEX_FILE)
    with open(DOCS_META, "w", encoding="utf-8") as f:
        json.dump({"docs": docs, "names": doc_names}, f)

# ---- Load index at startup ----
@app.on_event("startup")
def load_index():
    global index, docs, doc_names
    if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_META):
        index = faiss.read_index(INDEX_FILE)
        with open(DOCS_META, "r", encoding="utf-8") as f:
            data = json.load(f)
            docs = data["docs"]
            doc_names = data["names"]
    else:
        build_index()

# ---- Query endpoint ----
@app.post("/query")
def query(req: QueryRequest):
    global index, docs, doc_names
    if index.ntotal == 0:
        return {"answer": "No documents indexed yet."}

    vec = embedder.encode([req.question], convert_to_numpy=True)
    D, I = index.search(vec, k=1)
    idx = I[0][0]
    if idx >= len(docs):
        return {"answer": "No match found."}
    return {
        "answer": docs[idx],
        "source": doc_names[idx],
        "score": float(D[0][0])
    }

@app.get("/")
def root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

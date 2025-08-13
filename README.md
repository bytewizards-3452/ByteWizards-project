# Healthcare RAG Q&A App

## Features
- Healthcare Mode using preloaded FAISS index
- Custom Mode for user document upload
- FastAPI backend with HTML/JS frontend
- Drag & drop healthcare files into `/data/healthcare_docs/`

## Running locally
```bash
pip install -r requirements.txt
python main.py
```
Then open `http://127.0.0.1:8000` in your browser.

## Deploy on Replit
1. Import this repo into Replit using "GitHub → Import"
2. In Replit shell:
```bash
pip install -r requirements.txt
```
3. Click Run → App is live at `https://<your-app>.repl.co`

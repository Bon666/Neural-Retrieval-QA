# Neural Retrieval QA (FastAPI + FAISS + Bi-Encoder + Cross-Encoder)

A production-style retrieval QA pipeline:
1) Bi-Encoder retrieval (SentenceTransformer) -> topK candidates via FAISS
2) Cross-Encoder reranking -> best answer + debug top list

## Quickstart (local)
```bash
cd neural-retrieval-qa
python -m venv .venv
source .venv/bin/activate  # mac/linux
pip install -r requirements.txt

uvicorn src.app.main:app --reload --port 8000

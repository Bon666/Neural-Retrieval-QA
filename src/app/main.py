from __future__ import annotations
from fastapi import FastAPI
from .schemas import AskRequest, AskResponse
from .index import build_demo_index
from .retriever import BiEncoderRetriever
from .reranker import CrossEncoderReranker
from .pipeline import QAPipeline

app = FastAPI(title="Neural Retrieval QA", version="1.0.0")

_docs, _faiss_index, _embedder = build_demo_index()
_retriever = BiEncoderRetriever(docs=_docs, index=_faiss_index, embedder=_embedder)
_reranker = CrossEncoderReranker()
_pipe = QAPipeline(_retriever, _reranker)

@app.get("/health")
def health():
    return {"status": "ok", "docs": len(_docs)}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    out = _pipe.answer(req.query, k_retrieve=req.k_retrieve)
    return out

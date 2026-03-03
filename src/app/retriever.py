from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .index import Doc
from .settings import settings

@dataclass
class Candidate:
    doc_id: str
    text: str
    retriever_score: float
    reranker_score: float | None = None

class BiEncoderRetriever:
    def __init__(self, docs: List[Doc], index: faiss.Index, embedder: SentenceTransformer):
        self.docs = docs
        self.index = index
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 50) -> List[Candidate]:
        q_emb = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=settings.normalize_embeddings,
            show_progress_bar=False,
        ).astype("float32")

        k = min(k, len(self.docs))
        scores, idxs = self.index.search(q_emb, k)  # shapes: (1,k)
        scores = scores[0].tolist()
        idxs = idxs[0].tolist()

        out: List[Candidate] = []
        for s, i in zip(scores, idxs):
            if i < 0:
                continue
            d = self.docs[i]
            out.append(Candidate(doc_id=d.doc_id, text=d.text, retriever_score=float(s)))
        return out

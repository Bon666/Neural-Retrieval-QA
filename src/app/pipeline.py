from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
from .retriever import BiEncoderRetriever, Candidate
from .reranker import CrossEncoderReranker

class QAPipeline:
    def __init__(self, retriever: BiEncoderRetriever, reranker: CrossEncoderReranker):
        self.retriever = retriever
        self.reranker = reranker

    def answer(self, query: str, k_retrieve: int = 50) -> Dict[str, Any]:
        cands: List[Candidate] = self.retriever.retrieve(query, k=k_retrieve)
        if not cands:
            return {
                "query": query,
                "best": {
                    "doc_id": "",
                    "text": "",
                    "retriever_score": 0.0,
                    "reranker_score": None,
                },
                "debug_top": [],
            }

        pairs: List[Tuple[str, str]] = [(query, c.text) for c in cands]
        rr_scores = self.reranker.score(pairs)

        for c, s in zip(cands, rr_scores):
            c.reranker_score = float(s)

        best = max(cands, key=lambda x: x.reranker_score if x.reranker_score is not None else -1e9)

        debug_sorted = sorted(
            cands,
            key=lambda x: x.reranker_score if x.reranker_score is not None else -1e9,
            reverse=True,
        )[:10]

        return {
            "query": query,
            "best": {
                "doc_id": best.doc_id,
                "text": best.text,
                "retriever_score": best.retriever_score,
                "reranker_score": best.reranker_score,
            },
            "debug_top": [
                {
                    "doc_id": c.doc_id,
                    "retriever_score": c.retriever_score,
                    "reranker_score": c.reranker_score,
                }
                for c in debug_sorted
            ],
        }

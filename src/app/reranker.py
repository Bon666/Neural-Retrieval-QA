from __future__ import annotations
from typing import List, Tuple
import numpy as np
from sentence_transformers import CrossEncoder
from .settings import settings

class CrossEncoderReranker:
    """
    Cross-encoder scoring for (query, doc_text) pairs.
    Lazy-load to avoid heavy init at import time.
    """
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.cross_encoder_model
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        # CrossEncoder.predict returns list/np array; force np.float32
        scores = self.model.predict(pairs)
        return np.asarray(scores, dtype=np.float32)

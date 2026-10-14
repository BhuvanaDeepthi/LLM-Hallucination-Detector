import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import config

class EmbeddingManager:
    """Manages sentence embedding generation using Sentence-Transformers (MiniLM)."""

    _instance = None
    _model = None

    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        if EmbeddingManager._model is None:
            # Load SentenceTransformer model (all-MiniLM-L6-v2)
            EmbeddingManager._model = SentenceTransformer(self.model_name)

    @property
    def model(self) -> SentenceTransformer:
        if EmbeddingManager._model is None:
            self._ensure_model_loaded()
        return EmbeddingManager._model

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Generates dense vector embeddings for input text(s).
        Returns a float32 numpy array of shape (N, dimension).
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, config.EMBEDDING_DIMENSION), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return embeddings.astype(np.float32)

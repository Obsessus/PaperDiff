"""Vector Embedding Engine.

Provides:
- Embedding model initialization (SentenceTransformers with lazy loading & caching)
- Dense vector generation for document chunks and user queries
- NumPy-accelerated cosine similarity and FAISS vector index structures
"""

import os
import logging
from typing import List, Optional, Union, Tuple
import numpy as np
from src.rag.ingest import DocumentChunk

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Global cache for the loaded embedding model
_EMBEDDER_INSTANCE = None


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Lazy loader for SentenceTransformer model to optimize startup time and memory."""
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {model_name}")
            _EMBEDDER_INSTANCE = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer ({model_name}): {e}")
            raise RuntimeError(
                f"Could not load embedding model '{model_name}'. Ensure sentence-transformers is installed: {e}"
            )
    return _EMBEDDER_INSTANCE


class VectorIndex:
    """In-memory Vector Store for document chunks with normalized cosine similarity search."""

    def __init__(self, paper_id: str, paper_name: str):
        self.paper_id = paper_id
        self.paper_name = paper_name
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._faiss_index = None

    def add_chunks(self, chunks: List[DocumentChunk], model_name: str = DEFAULT_EMBEDDING_MODEL):
        """Compute embeddings and populate the index with chunks."""
        if not chunks:
            return

        self.chunks.extend(chunks)
        texts = [chunk.text for chunk in chunks]
        
        embedder = get_embedding_model(model_name)
        new_embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        new_embeddings = np.array(new_embeddings, dtype=np.float32)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        # Attempt to build FAISS index if faiss is available, else fallback to numpy dot product
        self._try_build_faiss()

    def _try_build_faiss(self):
        """Build FAISS IndexFlatIP (Inner Product = Cosine Similarity since vectors are L2 normalized)."""
        try:
            import faiss
            dim = self.embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(self.embeddings)
        except Exception:
            self._faiss_index = None

    def search(
        self,
        query: str,
        top_k: int = 4,
        model_name: str = DEFAULT_EMBEDDING_MODEL
    ) -> List[Tuple[DocumentChunk, float]]:
        """Search the most semantically relevant chunks for a query.

        Args:
            query: User or topic prompt text
            top_k: Maximum number of relevant chunks to return
            model_name: Embedding model identifier

        Returns:
            List of (DocumentChunk, similarity_score) sorted descending by relevance score.
        """
        if not self.chunks or self.embeddings is None or len(self.embeddings) == 0:
            return []

        embedder = get_embedding_model(model_name)
        query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype=np.float32)

        k = min(top_k, len(self.chunks))

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query_vec, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.chunks):
                    results.append((self.chunks[idx], float(score)))
            return results
        else:
            # High-performance NumPy Cosine similarity
            # Since both vectors are L2-normalized, inner product equals cosine similarity
            sim_scores = np.dot(self.embeddings, query_vec.T).flatten()
            top_indices = np.argsort(sim_scores)[::-1][:k]
            
            results = [
                (self.chunks[idx], float(sim_scores[idx]))
                for idx in top_indices
            ]
            return results

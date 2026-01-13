"""
Local Sentence-Transformer Embedding Client
Uses MiniLM for fast, local embeddings without Azure quota restrictions.
"""

import logging
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

from app.rag.embedding_base import EmbeddingClient

logger = logging.getLogger(__name__)


class LocalEmbeddingClient(EmbeddingClient):
    """Local embedding provider using sentence-transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize local embedding model.
        
        Args:
            model_name: Hugging Face model identifier
        """
        logger.info(f"Initializing local embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Local embedding model loaded. Dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts using local model.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings (float32)
        """
        try:
            # Generate embeddings
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            # Ensure float32 dtype
            return embeddings.astype("float32")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Return embedding dimension"""
        return self.embedding_dim
    
    def get_provider_name(self) -> str:
        """Return provider name"""
        return f"Local ({self.model_name})"

"""
Base Embedding Client Interface
Abstract base class for embedding providers.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class EmbeddingClient(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings (float32)
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Return the dimension of embedding vectors"""
        raise NotImplementedError
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the embedding provider"""
        raise NotImplementedError

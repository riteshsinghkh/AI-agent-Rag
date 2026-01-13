"""
Azure OpenAI Embedding Client
Azure OpenAI embedding provider (requires active quota).
"""

import os
import logging
import numpy as np
from typing import List
from openai import AzureOpenAI
from dotenv import load_dotenv

from app.rag.embedding_base import EmbeddingClient

load_dotenv()
logger = logging.getLogger(__name__)


class AzureEmbeddingClient(EmbeddingClient):
    """Azure OpenAI embedding provider"""
    
    def __init__(self):
        """Initialize Azure OpenAI embedding client"""
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        self.embedding_dim = 1536  # text-embedding-3-small dimension
        logger.info(f"Initialized Azure OpenAI embedding client with deployment: {self.deployment}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts using Azure OpenAI.
        
        Args:
            texts: List of text strings
            
        Returns:
            NumPy array of embeddings (float32)
        """
        try:
            # Clean texts
            cleaned_texts = [t.replace("\n", " ").strip() for t in texts if t.strip()]
            
            if not cleaned_texts:
                return np.array([], dtype="float32")
            
            # Process in batches (Azure OpenAI limit is typically 16 per request)
            batch_size = 16
            all_embeddings = []
            
            for i in range(0, len(cleaned_texts), batch_size):
                batch = cleaned_texts[i:i + batch_size]
                
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.deployment
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Generated embeddings for batch {i // batch_size + 1}")
            
            # Convert to numpy array
            embeddings_array = np.array(all_embeddings, dtype="float32")
            return embeddings_array
            
        except Exception as e:
            logger.error(f"Error generating Azure embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Return embedding dimension"""
        return self.embedding_dim
    
    def get_provider_name(self) -> str:
        """Return provider name"""
        return f"Azure OpenAI ({self.deployment})"

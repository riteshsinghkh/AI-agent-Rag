"""
Embedding Client Factory
Creates the appropriate embedding client based on configuration.
"""

import os
import logging
from dotenv import load_dotenv

from app.rag.embedding_base import EmbeddingClient

load_dotenv()
logger = logging.getLogger(__name__)


def get_embedding_client() -> EmbeddingClient:
    """
    Factory function to get the appropriate embedding client.
    
    Selects provider based on EMBEDDING_PROVIDER environment variable:
    - "local" → Local SentenceTransformer (default)
    - "azure" → Azure OpenAI embeddings (requires quota)
    
    Returns:
        EmbeddingClient instance
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    
    if provider == "local":
        try:
            logger.info("Using local SentenceTransformer for embeddings")
            # 🔽 DELAYED IMPORT (REQUIRED FIX)
            from app.rag.local_embedding_client import LocalEmbeddingClient
            return LocalEmbeddingClient()
        except ImportError as e:
            logger.error(f"Local embeddings unavailable: {e}")
            raise RuntimeError(
                "Local embedding provider selected but sentence-transformers is not installed"
            )

    elif provider == "azure":
        logger.info("Using Azure OpenAI for embeddings")
        # Import here to avoid circular dependencies
        from app.rag.azure_embedding_client import AzureEmbeddingClient
        return AzureEmbeddingClient()

    else:
        logger.warning(f"Unknown embedding provider '{provider}', defaulting to azure")
        from app.rag.azure_embedding_client import AzureEmbeddingClient
        return AzureEmbeddingClient()

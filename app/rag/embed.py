"""
Embedding Generation
Generates embeddings using Azure OpenAI.
"""

import os
from typing import List
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class EmbeddingGenerator:
    """Generates embeddings using Azure OpenAI"""
    
    def __init__(self):
        """Initialize Azure OpenAI client"""
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        # Validate configuration
        if not self.api_key or not self.endpoint:
            print("Warning: Azure OpenAI credentials not configured.")
            print("Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in .env file")
            self.client = None
        else:
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint
            )
            print(f"Embedding generator initialized with deployment: {self.deployment}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector (list of floats)
        """
        if self.client is None:
            raise ValueError("Azure OpenAI client not initialized. Check your credentials.")
        
        # Clean the text
        text = text.replace("\n", " ").strip()
        
        if not text:
            raise ValueError("Cannot generate embedding for empty text")
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.deployment
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts (batch processing)
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if self.client is None:
            raise ValueError("Azure OpenAI client not initialized. Check your credentials.")
        
        if not texts:
            return []
        
        # Clean texts
        cleaned_texts = [t.replace("\n", " ").strip() for t in texts]
        
        # Remove empty texts
        valid_texts = [(i, t) for i, t in enumerate(cleaned_texts) if t]
        
        if not valid_texts:
            return []
        
        embeddings = [None] * len(texts)
        
        # Process in batches (Azure OpenAI limit is typically 16 per request)
        batch_size = 16
        
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            batch_texts = [t[1] for t in batch]
            batch_indices = [t[0] for t in batch]
            
            try:
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=self.deployment
                )
                
                # Map embeddings back to original indices
                for j, embedding_data in enumerate(response.data):
                    original_index = batch_indices[j]
                    embeddings[original_index] = embedding_data.embedding
                
                print(f"Generated embeddings for batch {i // batch_size + 1}")
                
            except Exception as e:
                print(f"Error generating batch embeddings: {e}")
                raise
        
        # Replace None with empty list for any failed embeddings
        embeddings = [e if e is not None else [] for e in embeddings]
        
        return embeddings


# Global embedding generator instance (lazy initialization)
_embedder = None


def get_embedder() -> EmbeddingGenerator:
    """Get or create the global embedding generator instance"""
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingGenerator()
    return _embedder


# Convenience functions
def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a single text using global embedder"""
    return get_embedder().generate_embedding(text)


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts using global embedder"""
    return get_embedder().generate_embeddings(texts)

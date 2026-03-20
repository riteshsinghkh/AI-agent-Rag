"""
FAISS Vector Store
Manages FAISS index for similarity search.
"""

import faiss
import numpy as np
from typing import List, Dict, Optional
import pickle
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for document chunks"""
    
    def __init__(self, dimension: int = 384):
        """
        Initialize FAISS index
        
        Args:
            dimension: Dimension of embedding vectors 
                      (384 for all-MiniLM-L6-v2, 1536 for text-embedding-3-small)
        """
        self.dimension = dimension
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[str] = []  # Store original text chunks
        self.metadata: List[Dict] = []  # Store metadata (source filenames, etc.)
        self._is_initialized = False
    
    def create_index(self, embeddings: List[List[float]], chunks: List[str], metadata: List[Dict]):
        """
        Create FAISS index from embeddings
        
        Args:
            embeddings: List of embedding vectors
            chunks: List of text chunks
            metadata: List of metadata dicts for each chunk
        """
        if not embeddings:
            raise ValueError("No embeddings provided")
        
        if len(embeddings) != len(chunks) or len(embeddings) != len(metadata):
            raise ValueError("Embeddings, chunks, and metadata must have the same length")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Verify dimensions
        if embeddings_array.shape[1] != self.dimension:
            logger.warning(
                "Embedding dimension %s differs from expected %s; using detected value",
                embeddings_array.shape[1],
                self.dimension,
            )
            self.dimension = embeddings_array.shape[1]
        
        # Create FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add vectors to index
        self.index.add(embeddings_array)
        
        # Store chunks and metadata
        self.chunks = list(chunks)
        self.metadata = list(metadata)
        self._is_initialized = True
        
        logger.info("Created FAISS index with %s vectors (dim=%s)", len(embeddings), self.dimension)
    
    def add_embeddings(self, embeddings: List[List[float]], chunks: List[str], metadata: List[Dict]):
        """
        Add more embeddings to an existing index
        
        Args:
            embeddings: List of embedding vectors
            chunks: List of text chunks
            metadata: List of metadata dicts
        """
        if not self._is_initialized:
            self.create_index(embeddings, chunks, metadata)
            return
        
        # Convert to numpy
        embeddings_array = np.array(embeddings, dtype=np.float32)

        if embeddings_array.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings_array.shape[1]} does not match index dimension {self.dimension}"
            )
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Extend chunks and metadata
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)
        
        logger.info("Added %s vectors to FAISS index. Total vectors: %s", len(embeddings), self.index.ntotal)
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict]:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with chunk, metadata, and score
        """
        if not self._is_initialized or self.index is None:
            raise ValueError("Vector store not initialized. Call create_index first.")
        
        # Convert query to numpy array
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Limit top_k to available vectors
        top_k = min(top_k, self.index.ntotal)
        
        if top_k == 0:
            return []
        
        # Search
        distances, indices = self.index.search(query_array, top_k)
        
        # Build results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:  # FAISS returns -1 for invalid results
                continue
                
            results.append({
                "chunk": self.chunks[idx],
                "metadata": self.metadata[idx],
                "score": float(distance),  # L2 distance (lower is better)
                "rank": i + 1
            })
        
        return results
    
    def save(self, filepath: str = None):
        """
        Save index and data to disk
        
        Args:
            filepath: Path to save file (default: faiss_index.pkl in project root)
        """
        if not self._is_initialized:
            raise ValueError("Nothing to save. Vector store not initialized.")
        
        if filepath is None:
            # Default path in project root
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            filepath = project_root / "faiss_index.pkl"
        filepath = Path(filepath)
        
        # Prepare data for saving
        save_data = {
            "dimension": self.dimension,
            "chunks": self.chunks,
            "metadata": self.metadata,
            "index_data": faiss.serialize_index(self.index)
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)
        
        logger.info("Saved vector store to %s", filepath)
    
    def load(self, filepath: str = None) -> bool:
        """
        Load index and data from disk
        
        Args:
            filepath: Path to saved file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if filepath is None:
            # Default path in project root
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            filepath = project_root / "faiss_index.pkl"
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.info("No saved index found at %s", filepath)
            return False
        
        try:
            with open(filepath, "rb") as f:
                save_data = pickle.load(f)
            
            self.dimension = save_data["dimension"]
            self.chunks = save_data["chunks"]
            self.metadata = save_data["metadata"]
            self.index = faiss.deserialize_index(save_data["index_data"])
            self._is_initialized = True
            
            logger.info("Loaded vector store from %s", filepath)
            logger.info("Vector store stats: vectors=%s, dimension=%s", self.index.ntotal, self.dimension)
            return True
            
        except (FileNotFoundError, pickle.UnpicklingError, ValueError) as e:
            logger.error("Error loading vector store: %s", e)
            return False
    
    @property
    def is_initialized(self) -> bool:
        """Check if vector store is initialized"""
        return self._is_initialized
    
    @property
    def size(self) -> int:
        """Get number of vectors in the index"""
        if self.index is None:
            return 0
        return self.index.ntotal


# Global vector store instance (lazy initialization)
_vector_store = None


def get_vector_store(dimension: int = None) -> VectorStore:
    """
    Get or create the global vector store instance
    
    Args:
        dimension: Optional embedding dimension. If provided on first call, sets the dimension.
                  If not provided, uses default (384 for local embeddings).
    
    Returns:
        VectorStore instance
    """
    global _vector_store
    if _vector_store is None:
        if dimension is not None:
            _vector_store = VectorStore(dimension=dimension)
        else:
            # Default to 384 (local MiniLM model)
            _vector_store = VectorStore(dimension=384)
    return _vector_store

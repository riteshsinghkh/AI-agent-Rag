"""
Document Retriever
High-level interface for searching documents.
"""

from typing import List, Dict
from app.rag.embedding_factory import get_embedding_client
from app.rag.vectorstore import get_vector_store, VectorStore
from app.rag.documents import load_and_chunk_documents


# Default settings
DEFAULT_TOP_K = 3


def initialize_rag(docs_dir: str = None, force_rebuild: bool = False) -> VectorStore:
    """
    Initialize the RAG system by loading documents, creating embeddings, and building the index.
    
    Args:
        docs_dir: Directory containing documents (default: data/docs)
        force_rebuild: If True, rebuild index even if one exists on disk
        
    Returns:
        Initialized VectorStore instance
    """
    # Get embedding client first to determine dimension
    embedding_client = get_embedding_client()
    dimension = embedding_client.get_embedding_dimension()
    
    # Get or create vector store with correct dimension
    vector_store = get_vector_store(dimension=dimension)
    
    # Try to load existing index first (unless force_rebuild)
    if not force_rebuild and vector_store.load():
        print("Loaded existing index from disk")
        return vector_store
    
    print("Building new index from documents...")
    
    # Load and chunk documents
    print("\n1. Loading and chunking documents...")
    chunks = load_and_chunk_documents(docs_dir)
    
    if not chunks:
        raise ValueError("No documents found to index")
    
    # Extract texts for embedding
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings
    print(f"\n2. Generating embeddings for {len(texts)} chunks...")
    embedding_client = get_embedding_client()
    embeddings_array = embedding_client.embed_texts(texts)
    embeddings = embeddings_array.tolist()  # Convert numpy to list for vectorstore
    
    # Prepare metadata
    metadata = [{"source": chunk["source"], "chunk_index": chunk["chunk_index"]} for chunk in chunks]
    
    # Create index
    print("\n3. Creating FAISS index...")
    vector_store.create_index(embeddings, texts, metadata)
    
    # Save index to disk
    print("\n4. Saving index to disk...")
    vector_store.save()
    
    print("\nRAG system initialized successfully!")
    return vector_store


def search_documents(query: str, top_k: int = None) -> List[Dict]:
    """
    Search for relevant documents based on query
    
    Args:
        query: User's search query
        top_k: Number of documents to retrieve (default: 3)
        
    Returns:
        List of dictionaries containing:
        - chunk: Text chunk
        - source: Source filename
        - score: Similarity score (lower is better for L2)
    """
    top_k = top_k or DEFAULT_TOP_K
    
    vector_store = get_vector_store()
    
    # Check if index is initialized
    if not vector_store.is_initialized:
        # Try to load from disk
        if not vector_store.load():
            raise ValueError("Vector store not initialized. Call initialize_rag() first.")
    
    # Generate query embedding
    embedding_client = get_embedding_client()
    query_embedding_array = embedding_client.embed_texts([query])
    query_embedding = query_embedding_array[0].tolist()  # Convert to list
    
    # Search
    results = vector_store.search(query_embedding, top_k)
    
    # Search
    results = vector_store.search(query_embedding, top_k)
    
    # Format results
    formatted_results = []
    for result in results:
        formatted_results.append({
            "chunk": result["chunk"],
            "source": result["metadata"]["source"],
            "chunk_index": result["metadata"]["chunk_index"],
            "score": result["score"]
        })
    
    return formatted_results


def get_unique_sources(results: List[Dict]) -> List[str]:
    """
    Extract unique source filenames from search results
    
    Args:
        results: Search results from search_documents
        
    Returns:
        List of unique source filenames
    """
    sources = []
    seen = set()
    
    for result in results:
        source = result.get("source", "")
        if source and source not in seen:
            sources.append(source)
            seen.add(source)
    
    return sources


def format_context_for_llm(results: List[Dict]) -> str:
    """
    Format search results into a context string for the LLM
    
    Args:
        results: Search results from search_documents
        
    Returns:
        Formatted context string
    """
    if not results:
        return "No relevant documents found."
    
    context_parts = []
    for i, result in enumerate(results, 1):
        source = result["source"]
        chunk = result["chunk"]
        context_parts.append(f"[Document {i}: {source}]\n{chunk}")
    
    return "\n\n---\n\n".join(context_parts)

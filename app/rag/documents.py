"""
Document Loader and Chunking
Loads documents from disk and splits them into chunks.
"""

import os
from typing import List, Dict
from pathlib import Path

from app.rag.parsing import parse_document, SUPPORTED_EXTENSIONS


# Default settings (can be overridden)
DEFAULT_CHUNK_SIZE = 400  # tokens (approx)
DEFAULT_CHUNK_OVERLAP = 50  # tokens (approx)
CHARS_PER_TOKEN = 4  # Approximate: 1 token ≈ 4 characters


def load_documents(docs_dir: str = None) -> List[Dict[str, str]]:
    """
    Load documents from the docs directory
    
    Args:
        docs_dir: Directory containing document files (default: data/docs)
        
    Returns:
        List of dictionaries with 'filename' and 'content'
    """
    # Determine docs directory path
    if docs_dir is None:
        # Get the project root (where app/ folder is)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        docs_dir = project_root / "data" / "docs"
    else:
        docs_dir = Path(docs_dir)
    
    documents = []
    
    # Check if directory exists
    if not docs_dir.exists():
        print(f"Warning: Documents directory not found: {docs_dir}")
        return documents
    
    # Find all supported files in the directory
    files = [p for p in docs_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    
    if not files:
        print(f"Warning: No files found in {docs_dir}")
        return documents
    
    # Read each file as text when possible
    for file_path in files:
        content = parse_document(file_path)

        if not content:
            print(f"Skipping empty or unreadable file {file_path.name}")
            continue

        documents.append({
            "filename": file_path.name,
            "content": content
        })
        print(f"Loaded: {file_path.name} ({len(content)} chars)")
    
    print(f"Total documents loaded: {len(documents)}")
    return documents


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Split text into overlapping chunks based on token count (approximated)
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in tokens (default: 400)
        overlap: Overlap between chunks in tokens (default: 50)
        
    Returns:
        List of text chunks
    """
    chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
    overlap = overlap or DEFAULT_CHUNK_OVERLAP
    
    # Convert token counts to character counts
    chunk_chars = chunk_size * CHARS_PER_TOKEN
    overlap_chars = overlap * CHARS_PER_TOKEN
    
    # Clean the text (normalize whitespace)
    text = " ".join(text.split())
    
    if len(text) <= chunk_chars:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position
        end = start + chunk_chars
        
        # If not at the end, try to break at a sentence or word boundary
        if end < len(text):
            # Look for sentence boundary (., !, ?) within last 100 chars
            search_start = max(end - 100, start)
            last_period = text.rfind(".", search_start, end)
            last_question = text.rfind("?", search_start, end)
            last_exclaim = text.rfind("!", search_start, end)
            
            # Find the latest sentence boundary
            boundary = max(last_period, last_question, last_exclaim)
            
            if boundary > start:
                end = boundary + 1
            else:
                # No sentence boundary, look for word boundary (space)
                last_space = text.rfind(" ", search_start, end)
                if last_space > start:
                    end = last_space
        
        # Extract chunk
        chunk = text[start:end].strip()
        
        if chunk:
            chunks.append(chunk)
        
        # Move start position (with overlap)
        start = end - overlap_chars
        
        # Prevent infinite loop
        if start >= len(text) - overlap_chars:
            break
    
    return chunks


def chunk_documents(documents: List[Dict[str, str]], 
                    chunk_size: int = None, 
                    overlap: int = None) -> List[Dict]:
    """
    Chunk all documents and return chunks with metadata
    
    Args:
        documents: List of documents from load_documents()
        chunk_size: Size of each chunk in tokens
        overlap: Overlap between chunks in tokens
        
    Returns:
        List of dictionaries with 'text', 'source', 'chunk_index'
    """
    all_chunks = []
    
    for doc in documents:
        filename = doc["filename"]
        content = doc["content"]
        
        # Chunk the document
        text_chunks = chunk_text(content, chunk_size, overlap)
        
        # Add metadata to each chunk
        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "text": chunk,
                "source": filename,
                "chunk_index": i
            })
        
        print(f"  {filename}: {len(text_chunks)} chunks")
    
    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


# Convenience function to load and chunk in one step
def load_and_chunk_documents(docs_dir: str = None, 
                              chunk_size: int = None, 
                              overlap: int = None) -> List[Dict]:
    """
    Load documents and chunk them in one step
    
    Args:
        docs_dir: Directory containing document files
        chunk_size: Size of each chunk in tokens
        overlap: Overlap between chunks in tokens
        
    Returns:
        List of chunk dictionaries with text, source, chunk_index
    """
    documents = load_documents(docs_dir)
    chunks = chunk_documents(documents, chunk_size, overlap)
    return chunks

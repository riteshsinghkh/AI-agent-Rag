"""
Test script for RAG pipeline
Run this script to verify that the RAG system is working correctly.

Usage:
    python test_rag.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def test_document_loading():
    """Test document loading and chunking"""
    print("\n" + "="*60)
    print("TEST 1: Document Loading and Chunking")
    print("="*60)
    
    from app.rag.documents import load_documents, chunk_documents
    
    # Load documents
    documents = load_documents()
    
    if not documents:
        print("❌ No documents loaded!")
        return False
    
    print(f"✅ Loaded {len(documents)} documents")
    
    # Chunk documents
    chunks = chunk_documents(documents)
    
    if not chunks:
        print("❌ No chunks created!")
        return False
    
    print(f"✅ Created {len(chunks)} chunks")
    
    # Show sample chunk
    print("\nSample chunk:")
    print(f"  Source: {chunks[0]['source']}")
    print(f"  Text: {chunks[0]['text'][:200]}...")
    
    return True


def test_embedding_generation():
    """Test embedding generation (requires Azure OpenAI credentials)"""
    print("\n" + "="*60)
    print("TEST 2: Embedding Generation")
    print("="*60)
    
    from app.rag.embed import get_embedder
    
    embedder = get_embedder()
    
    if embedder.client is None:
        print("⚠️  Azure OpenAI not configured. Skipping embedding test.")
        print("   Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in .env")
        return None  # Skip, not fail
    
    # Test single embedding
    test_text = "What is the company leave policy?"
    
    try:
        embedding = embedder.generate_embedding(test_text)
        print(f"✅ Generated embedding with {len(embedding)} dimensions")
        print(f"   First 5 values: {embedding[:5]}")
        return True
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return False


def test_full_rag_pipeline():
    """Test the complete RAG pipeline"""
    print("\n" + "="*60)
    print("TEST 3: Full RAG Pipeline")
    print("="*60)
    
    from app.rag.retriever import initialize_rag, search_documents, get_unique_sources
    
    # Check if Azure OpenAI is configured
    from app.rag.embed import get_embedder
    embedder = get_embedder()
    
    if embedder.client is None:
        print("⚠️  Azure OpenAI not configured. Skipping full RAG test.")
        return None
    
    try:
        # Initialize RAG system
        print("\nInitializing RAG system...")
        initialize_rag(force_rebuild=True)
        
        # Test queries
        test_queries = [
            "What is the leave policy?",
            "Can I work remotely?",
            "How do I expense a business lunch?",
            "What are the security requirements for passwords?"
        ]
        
        print("\n" + "-"*40)
        print("Testing search queries:")
        print("-"*40)
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            results = search_documents(query, top_k=2)
            
            if results:
                sources = get_unique_sources(results)
                print(f"  Found {len(results)} results from: {sources}")
                print(f"  Top result score: {results[0]['score']:.4f}")
                print(f"  Top result snippet: {results[0]['chunk'][:100]}...")
            else:
                print("  No results found")
        
        print("\n✅ Full RAG pipeline test complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error in RAG pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("RAG PIPELINE TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Test 1: Document loading
    results["Document Loading"] = test_document_loading()
    
    # Test 2: Embedding generation
    results["Embedding Generation"] = test_embedding_generation()
    
    # Test 3: Full pipeline
    results["Full RAG Pipeline"] = test_full_rag_pipeline()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  SKIPPED"
        print(f"  {test_name}: {status}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

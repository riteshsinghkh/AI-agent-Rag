"""
Test Script for AI Agent
Tests the agent decision logic, tool calling, and session memory.

Run: python test_agent.py

IMPORTANT: Make sure to:
1. Initialize RAG first: python test_rag.py
2. Configure .env with valid Azure OpenAI credentials
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_direct_answer():
    """Test 1: Direct answers without document search"""
    print("\n" + "="*60)
    print("TEST 1: Direct Answer (No Documents Needed)")
    print("="*60)
    
    from app.agent.agent import ask
    
    # Questions that should be answered directly
    test_queries = [
        "What is 2+2?",
        "Hello!",
        "What is the capital of France?",
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        answer, sources, chunks, confidence = ask(query)
        print(f"💬 Answer: {answer[:200]}..." if len(answer) > 200 else f"💬 Answer: {answer}")
        print(f"📂 Sources: {sources}")
        
        # Direct answers should have no sources
        if not sources:
            print("✅ PASS: No documents used (as expected)")
        else:
            print("⚠️  NOTE: Documents were used (may be acceptable)")


def test_document_retrieval():
    """Test 2: Questions requiring document search"""
    print("\n" + "="*60)
    print("TEST 2: Document Retrieval (RAG)")
    print("="*60)
    
    from app.agent.agent import ask
    
    # Questions that should trigger document search
    test_queries = [
        "What is the leave policy?",
        "How many vacation days do I get?",
        "Can I work remotely?",
        "What is the password policy?",
        "How do I submit expense reports?",
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        answer, sources, chunks, confidence = ask(query)
        print(f"💬 Answer: {answer[:300]}..." if len(answer) > 300 else f"💬 Answer: {answer}")
        print(f"📂 Sources: {sources}")
        
        # Document queries should have sources
        if sources:
            print("✅ PASS: Documents retrieved successfully")
        else:
            print("⚠️  WARNING: No documents found - check RAG initialization")


def test_session_memory():
    """Test 3: Session memory for follow-up questions"""
    print("\n" + "="*60)
    print("TEST 3: Session Memory (Follow-up Questions)")
    print("="*60)
    
    from app.agent.agent import ask
    from app.agent.memory import memory
    
    session_id = "test-session-123"
    
    # Clear any existing session
    memory.clear_session(session_id)
    
    # Conversation with follow-up
    conversation = [
        "What is the leave policy?",
        "How many sick days are included?",
        "Can I carry over unused days?",
    ]
    
    for i, query in enumerate(conversation):
        print(f"\n📝 Query {i+1}: {query}")
        answer, sources, chunks, confidence = ask(query, session_id=session_id)
        print(f"💬 Answer: {answer[:300]}..." if len(answer) > 300 else f"💬 Answer: {answer}")
        print(f"📂 Sources: {sources}")
    
    # Check session history
    history = memory.get_history(session_id)
    print(f"\n📊 Session History Length: {len(history)} messages")
    
    if len(history) > 0:
        print("✅ PASS: Session memory is working")
    else:
        print("❌ FAIL: Session memory not storing messages")


def test_tool_detection():
    """Test 4: Verify tool call detection logic"""
    print("\n" + "="*60)
    print("TEST 4: Tool Call Detection")
    print("="*60)
    
    from app.agent.agent import Agent
    
    agent = Agent()
    
    # Test responses that should trigger tool call
    tool_responses = [
        "TOOL_CALL: search_documents",
        "Let me search for that.\nTOOL_CALL: search_documents",
        "I need to check the documents.\nTOOL_CALL: search_documents\nPlease wait.",
    ]
    
    # Test responses that should NOT trigger tool call
    direct_responses = [
        "The answer is 42.",
        "Hello! How can I help you?",
        "Based on my knowledge, the capital of France is Paris.",
    ]
    
    print("\n🔍 Testing tool call detection:")
    
    for response in tool_responses:
        detected = agent._needs_tool_call(response)
        status = "✅" if detected else "❌"
        print(f"{status} '{response[:40]}...' -> Tool call: {detected}")
    
    print("\n🔍 Testing direct response detection:")
    
    for response in direct_responses:
        detected = agent._needs_tool_call(response)
        status = "✅" if not detected else "❌"
        print(f"{status} '{response[:40]}...' -> Tool call: {detected}")


def main():
    """Run all tests"""
    print("="*60)
    print("       AI AGENT TEST SUITE")
    print("="*60)
    print("\n⚠️  PREREQUISITES:")
    print("   1. Run 'python test_rag.py' first to initialize RAG")
    print("   2. Configure .env with Azure OpenAI credentials")
    
    # Check if RAG is initialized
    index_path = os.path.join(os.path.dirname(__file__), "faiss_index.pkl")
    if not os.path.exists(index_path):
        print("\n❌ ERROR: FAISS index not found!")
        print("   Please run 'python test_rag.py' first to build the index.")
        sys.exit(1)
    
    # Check for Azure credentials
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("\n❌ ERROR: Azure OpenAI credentials not configured!")
        print("   Please update .env with your Azure OpenAI credentials.")
        sys.exit(1)
    
    print("\n✅ Prerequisites check passed!")
    print("\nStarting tests...\n")
    
    try:
        # Test 4 first (doesn't need API calls)
        test_tool_detection()
        
        # Tests that need API calls
        test_direct_answer()
        test_document_retrieval()
        test_session_memory()
        
        print("\n" + "="*60)
        print("       ALL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

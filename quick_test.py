"""Quick test script to verify the agent works"""
import os
os.environ['LLM_PROVIDER'] = 'huggingface'
os.environ['EMBEDDING_PROVIDER'] = 'local'

print("=" * 70)
print("INITIALIZING RAG SYSTEM...")
print("=" * 70)

from app.rag.retriever import initialize_rag

try:
    initialize_rag()
    print("\n✅ RAG system initialized successfully!\n")
except Exception as e:
    print(f"\n❌ Error initializing RAG: {e}\n")
    import traceback
    traceback.print_exc()

print("=" * 70)
print("TESTING AGENT...")
print("=" * 70)

from app.agent.agent import get_agent

agent = get_agent()

# Test 1: Question that needs RAG
print("\n📝 Test 1: Question needing documents")
print("Question: 'What is the leave policy?'")
try:
    answer, sources = agent.process_query("What is the leave policy?", session_id="test1")
    print(f"\n✅ Answer: {answer[:200]}...")
    print(f"📄 Sources: {sources}")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: General question
print("\n" + "=" * 70)
print("\n📝 Test 2: General question")
print("Question: 'What is 2+2?'")
try:
    answer, sources = agent.process_query("What is 2+2?", session_id="test2")
    print(f"\n✅ Answer: {answer}")
    print(f"📄 Sources: {sources}")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ TESTING COMPLETE!")
print("=" * 70)

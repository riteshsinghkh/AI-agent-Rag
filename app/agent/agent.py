"""
Agent Decision Logic
Implements the core AI agent that decides whether to answer directly or use RAG.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

from app.agent.prompt import get_system_prompt, get_context_prompt, get_no_context_prompt
from app.agent.memory import memory
from app.rag.retriever import search_documents, format_context_for_llm, get_unique_sources
from app.llm.factory import get_llm_client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Agent:
    """AI Agent with decision-making and tool-calling capabilities"""
    
    def __init__(self):
        """Initialize the agent with LLM client"""
        self.llm_client = get_llm_client()
        self.system_prompt = get_system_prompt()
        logger.info(f"Agent initialized with LLM provider: {self.llm_client.get_provider_name()}")
    
    def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Call LLM with given messages (provider-agnostic)
        
        Args:
            messages: List of message dictionaries
            temperature: LLM temperature (0.0-1.0)
            
        Returns:
            LLM response content
        """
        try:
            return self.llm_client.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=1024
            )
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise
    
    def _needs_tool_call(self, response: str) -> bool:
        """
        Check if the LLM response indicates a tool call is needed
        
        Args:
            response: LLM response text
            
        Returns:
            True if tool call detected
        """
        return "TOOL_CALL: search_documents" in response
    
    def _build_messages(self, query: str, session_id: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Build message list with system prompt and conversation history
        
        Args:
            query: Current user query
            session_id: Optional session ID for history
            
        Returns:
            List of messages for LLM
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history if session exists
        if session_id:
            history = memory.get_history(session_id)
            messages.extend(history)
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def process_query(self, query: str, session_id: Optional[str] = None) -> Tuple[str, List[str]]:
        """
        Process user query and return answer with sources
        
        Args:
            query: User's question
            session_id: Optional session identifier for memory
            
        Returns:
            Tuple of (answer, list of source documents)
        """
        sources: List[str] = []
        
        try:
            # Step 1: Build messages with history
            messages = self._build_messages(query, session_id)
            logger.info(f"Processing query: {query[:50]}...")
            
            # Step 2: First LLM call - decide if tool is needed
            initial_response = self._call_llm(messages, temperature=0.3)
            logger.info(f"Initial response: {initial_response[:100]}...")
            
            # Step 3: Check if tool call is needed
            if self._needs_tool_call(initial_response):
                logger.info("Tool call detected - searching documents...")
                
                # Step 4: Call search_documents tool
                search_results = search_documents(query, top_k=3)
                
                if search_results:
                    # Format context for LLM
                    context = format_context_for_llm(search_results)
                    sources = get_unique_sources(search_results)
                    logger.info(f"Found {len(search_results)} chunks from {len(sources)} sources")
                    
                    # Step 5: Generate answer with context
                    context_prompt = get_context_prompt(context, query)
                    context_messages = [
                        {"role": "system", "content": context_prompt}
                    ]
                    final_answer = self._call_llm(context_messages, temperature=0.5)
                else:
                    # No documents found
                    logger.warning("No relevant documents found")
                    no_context_prompt = get_no_context_prompt(query)
                    context_messages = [
                        {"role": "system", "content": no_context_prompt}
                    ]
                    final_answer = self._call_llm(context_messages, temperature=0.5)
            else:
                # Direct answer without tool call
                logger.info("Answering directly without documents")
                final_answer = initial_response
            
            # Step 6: Update session memory
            if session_id:
                memory.add_message(session_id, "user", query)
                memory.add_message(session_id, "assistant", final_answer)
                logger.info(f"Updated session memory for: {session_id}")
            
            return final_answer, sources
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}", []


# Global agent instance (lazy initialization)
_agent: Optional[Agent] = None


def get_agent() -> Agent:
    """Get or create the global agent instance"""
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent


def ask(query: str, session_id: Optional[str] = None) -> Tuple[str, List[str]]:
    """
    Convenience function to process a query
    
    Args:
        query: User's question
        session_id: Optional session identifier
        
    Returns:
        Tuple of (answer, list of source documents)
    """
    agent = get_agent()
    return agent.process_query(query, session_id)

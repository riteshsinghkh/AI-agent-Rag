"""
Agent Decision Logic
Implements the core AI agent that decides whether to answer directly or use RAG.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv

from app.agent.prompt import get_system_prompt, get_context_prompt, get_no_context_prompt, get_structured_prompt
from app.agent.memory import memory
from app.rag.retriever import (
    search_documents,
    format_context_for_llm,
    get_unique_sources,
    get_max_confidence,
    has_index_data,
)
from app.config import settings
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

    def _format_structured_output(self, data: Dict[str, Any]) -> str:
        """Format structured JSON into readable text."""
        def normalize_label(value: str) -> str:
            return value.replace("_", " ").title()

        def format_scalar(value: Any) -> str:
            return "null" if value is None else str(value)

        def format_list_of_dicts(key: str, items: List[Dict[str, Any]]) -> str:
            singular = key[:-1] if key.endswith("s") and len(key) > 1 else "Item"
            singular_label = normalize_label(singular)
            parts = []
            for idx, item in enumerate(items, 1):
                parts.append(f"{singular_label} {idx}: {flatten_dict(item)}")
            return "; ".join(parts)

        def flatten_dict(values: Dict[str, Any]) -> str:
            parts = []
            for key, value in values.items():
                label = normalize_label(key)
                if isinstance(value, dict):
                    parts.append(f"{label}: {flatten_dict(value)}")
                elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                    parts.append(f"{label}: {format_list_of_dicts(key, value)}")
                elif isinstance(value, list):
                    items = ", ".join(format_scalar(item) for item in value) if value else "[]"
                    parts.append(f"{label}: {items}")
                else:
                    parts.append(f"{label}: {format_scalar(value)}")
            return " | ".join(parts)

        lines: List[str] = []
        for section, values in data.items():
            title = normalize_label(section)
            lines.append(f"**{title}:**")
            if isinstance(values, dict):
                for key, value in values.items():
                    label = normalize_label(key)
                    if isinstance(value, dict):
                        lines.append(f"- {label}: {flatten_dict(value)}")
                    elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                        lines.append(f"- {label}: {format_list_of_dicts(key, value)}")
                    elif isinstance(value, list):
                        items = ", ".join(format_scalar(item) for item in value) if value else "[]"
                        lines.append(f"- {label}: {items}")
                    else:
                        lines.append(f"- {label}: {format_scalar(value)}")
            elif isinstance(values, list) and values and all(isinstance(item, dict) for item in values):
                for idx, item in enumerate(values, 1):
                    lines.append(f"- Item {idx}: {flatten_dict(item)}")
            elif isinstance(values, list):
                for item in values:
                    lines.append(f"- {format_scalar(item)}")
            else:
                lines.append(f"- {format_scalar(values)}")
            lines.append("")
        return "\n".join(lines).strip()

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Attempt to parse JSON from LLM output."""
        if not text:
            return None

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None

        return payload if isinstance(payload, dict) else None

    def _try_structured_answer(self, context: str, query: str) -> Optional[str]:
        """Ask the LLM for structured JSON and format it deterministically."""
        structured_prompt = get_structured_prompt(context, query)
        messages = [{"role": "system", "content": structured_prompt}]
        response = self._call_llm(messages, temperature=0.2)
        parsed = self._parse_json_response(response)
        if not parsed:
            return None
        return self._format_structured_output(parsed)

    def process_query(self, query: str, session_id: Optional[str] = None) -> Tuple[str, List[str], List[Dict], Optional[float]]:
        """
        Process user query and return answer with sources
        
        Args:
            query: User's question
            session_id: Optional session identifier for memory
            
        Returns:
            Tuple of (answer, list of source documents, chunks, confidence)
        """
        sources: List[str] = []
        chunks: List[Dict] = []
        confidence: Optional[float] = None
        
        try:
            # Step 1: Build messages with history
            messages = self._build_messages(query, session_id)
            logger.info(f"Processing query: {query[:50]}...")
            
            # Step 2: First LLM call - decide if tool is needed
            initial_response = self._call_llm(messages, temperature=0.3)
            logger.info(f"Initial response: {initial_response[:100]}...")
            
            # Step 3: Decide if tool call is needed
            force_retrieval = has_index_data()
            use_retrieval = force_retrieval or self._needs_tool_call(initial_response)

            if use_retrieval:
                logger.info("Tool call detected - searching documents...")
                
                # Step 4: Call search_documents tool
                search_results = search_documents(query, top_k=settings.TOP_K)
                
                if search_results:
                    chunks = search_results
                    confidence = get_max_confidence(search_results)

                    if should_reject_results(search_results, settings.CONFIDENCE_THRESHOLD):
                        logger.warning("Confidence below threshold; returning not found")
                        final_answer = "Not found in document."
                    else:
                        # Format context for LLM
                        context = format_context_for_llm(search_results)
                        sources = get_unique_sources(search_results)
                        logger.info(f"Found {len(search_results)} chunks from {len(sources)} sources")

                        # Step 5: Generate structured answer with context
                        structured_answer = self._try_structured_answer(context, query)
                        if structured_answer:
                            final_answer = structured_answer
                        else:
                            context_prompt = get_context_prompt(context, query)
                            context_messages = [
                                {"role": "system", "content": context_prompt}
                            ]
                            final_answer = self._call_llm(context_messages, temperature=0.5)
                else:
                    # No documents found
                    logger.warning("No relevant documents found")
                    final_answer = "Not found in document."
            else:
                # Direct answer without tool call
                logger.info("Answering directly without documents")
                final_answer = initial_response
            
            # Step 6: Update session memory
            if session_id:
                memory.add_message(session_id, "user", query)
                memory.add_message(session_id, "assistant", final_answer)
                logger.info(f"Updated session memory for: {session_id}")
            
            return final_answer, sources, chunks, confidence
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}", [], [], None


def should_reject_results(results: List[Dict], threshold: float) -> bool:
    """Return True when results should be rejected by guardrails."""
    if not results:
        return True
    return get_max_confidence(results) < threshold


# Global agent instance (lazy initialization)
_agent: Optional[Agent] = None


def get_agent() -> Agent:
    """Get or create the global agent instance"""
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent


def ask(query: str, session_id: Optional[str] = None) -> Tuple[str, List[str], List[Dict], Optional[float]]:
    """
    Convenience function to process a query
    
    Args:
        query: User's question
        session_id: Optional session identifier
        
    Returns:
        Tuple of (answer, list of source documents, chunks, confidence)
    """
    agent = get_agent()
    return agent.process_query(query, session_id)

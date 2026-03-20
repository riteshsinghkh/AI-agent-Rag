"""
Agent Decision Logic
Implements the core AI agent that decides whether to answer directly or use RAG.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv

from app.agent.prompt import get_system_prompt, get_context_prompt, get_structured_prompt
from app.agent.memory import memory
from app.rag.retriever import (
    search_documents,
    format_context_for_llm,
    get_unique_sources,
    get_max_confidence,
    has_index_data,
    list_latest_uploaded_documents,
    get_all_chunks_for_source,
)
from app.config import settings
from app.llm.factory import get_llm_client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DESCRIBE_INTENTS = re.compile(
    r"\b(describe|summarize|summary|what'?s?\s+in|tell\s+me\s+about|show\s+me|explain|overview)\b",
    re.IGNORECASE,
)


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

    def _is_list_uploaded_documents_query(self, query: str) -> bool:
        """Detect requests that ask to list or describe ALL uploaded documents."""
        lowered = query.lower().strip()
        patterns = [
            r"\ball uploaded documents\b",
            r"\buploaded documents\b",
            r"\bwhat( are|'re| is)?\s+(all\s+)?the\s+uploaded\s+documents\b",
            r"\blist\s+(all\s+)?uploaded\s+(files|documents)\b",
            r"\bshow\s+(all\s+)?uploaded\s+(files|documents)\b",
            r"\bdescribe\s+(all\s+)?(the\s+)?uploaded\s+(files|documents)\b",
            r"\bwhat\s+(files|documents)\s+(have\s+been\s+|are\s+|were\s+)?uploaded\b",
            r"\bwhat\s+documents\s+(are|were)\s+uploaded\b",
        ]
        return any(re.search(pattern, lowered) for pattern in patterns)

    def _detect_target_document(self, query: str) -> Optional[str]:
        """Fuzzy-match query keywords against latest uploaded filenames. Returns filename or None."""
        documents = list_latest_uploaded_documents()
        if not documents:
            return None

        query_lower = query.lower()
        query_words = [w for w in re.split(r"\W+", query_lower) if len(w) > 2]

        best_match: Optional[str] = None
        best_score = 0

        for doc in documents:
            name = doc["name"]
            # Strip UUID suffix (32 hex chars) and extension, then normalize
            stem = Path(name).stem
            clean_stem = re.sub(r"-[0-9a-f]{32}$", "", stem, flags=re.IGNORECASE)
            clean_stem = re.sub(r"[\s_\-\(\)]+", " ", clean_stem).lower().strip()

            doc_words = [w for w in clean_stem.split() if len(w) > 2]
            # Forward: doc keywords found in query; reverse: query keywords found in doc name
            forward = sum(1 for w in doc_words if w in query_lower)
            reverse = sum(1 for w in query_words if w in clean_stem)
            score = forward + reverse

            if score > best_score:
                best_score = score
                best_match = name

        return best_match if best_score >= 1 else None

    def _detect_single_document_describe_query(self, query: str) -> Optional[str]:
        """Return matched filename when query asks to describe a specific document."""
        if not _DESCRIBE_INTENTS.search(query):
            return None
        return self._detect_target_document(query)

    def _build_single_document_answer(
        self, target_source: str, query: str
    ) -> Tuple[str, List[str], List[Dict], Optional[float]]:
        """Retrieve all indexed chunks for a specific document and generate an LLM description."""
        doc_chunks = get_all_chunks_for_source(target_source)
        if not doc_chunks:
            return f"No indexed content found for document: {target_source}", [], [], None

        context = f"[Document: {target_source}]\n" + "\n\n---\n\n".join(doc_chunks)
        context_prompt = get_context_prompt(context, query)
        messages = [{"role": "system", "content": context_prompt}]
        answer = self._call_llm(messages, temperature=0.5)
        return answer, [target_source], [], None

    def _build_uploaded_documents_answer(self) -> Tuple[str, List[str], List[Dict], Optional[float]]:
        """Generate LLM descriptions for all latest uploaded documents."""
        documents = list_latest_uploaded_documents()
        if not documents:
            return "No uploaded documents found.", [], [], None

        sources = [doc["name"] for doc in documents]

        # Build full context with all documents labelled
        context_parts = []
        for doc in documents:
            doc_chunks = get_all_chunks_for_source(doc["name"])
            if doc_chunks:
                context_parts.append(
                    f"[Document: {doc['name']}]\n" + "\n---\n".join(doc_chunks)
                )
            else:
                context_parts.append(
                    f"[Document: {doc['name']}]\n{doc.get('preview', 'No content available.')}"
                )

        context = "\n\n=====\n\n".join(context_parts)
        describe_query = (
            "For each of the uploaded documents listed above, provide: its name, purpose, "
            "and a clear summary of its key contents in 2-4 sentences."
        )
        context_prompt = get_context_prompt(context, describe_query)
        messages = [{"role": "system", "content": context_prompt}]

        try:
            answer = self._call_llm(messages, temperature=0.3)
        except Exception:
            # Fallback to simple list if LLM fails
            lines = ["Here are your uploaded documents:"]
            for idx, doc in enumerate(documents, 1):
                lines.append(f"\n{idx}. **{doc['name']}**\n   {doc['preview']}")
            answer = "\n".join(lines)

        return answer, sources, [], None

    def process_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        use_latest_uploads_only: Optional[bool] = None,
    ) -> Tuple[str, List[str], List[Dict], Optional[float]]:
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
            logger.info(f"Processing query: {query[:50]}...")

            # Fast path: avoid semantic top-k omission for document listing/description.
            if self._is_list_uploaded_documents_query(query):
                logger.info("Detected all-uploaded-documents query")
                final_answer, sources, chunks, confidence = self._build_uploaded_documents_answer()
            elif (target_doc := self._detect_single_document_describe_query(query)) is not None:
                logger.info("Detected single-document describe query for: %s", target_doc)
                final_answer, sources, chunks, confidence = self._build_single_document_answer(
                    target_doc, query
                )
            else:
                # Step 1: Build messages with history
                messages = self._build_messages(query, session_id)

                # Step 2: First LLM call - decide if tool is needed
                initial_response = self._call_llm(messages, temperature=0.3)
                logger.info(f"Initial response: {initial_response[:100]}...")

                # Step 3: Decide if tool call is needed
                force_retrieval = has_index_data()
                use_retrieval = force_retrieval or self._needs_tool_call(initial_response)

                if use_retrieval:
                    logger.info("Tool call detected - searching documents...")

                    # Step 4: Call search_documents tool
                    search_results = search_documents(
                        query,
                        top_k=settings.TOP_K,
                        restrict_to_latest_uploads=use_latest_uploads_only,
                    )

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


def ask(
    query: str,
    session_id: Optional[str] = None,
    use_latest_uploads_only: Optional[bool] = None,
) -> Tuple[str, List[str], List[Dict], Optional[float]]:
    """
    Convenience function to process a query
    
    Args:
        query: User's question
        session_id: Optional session identifier
        
    Returns:
        Tuple of (answer, list of source documents, chunks, confidence)
    """
    agent = get_agent()
    return agent.process_query(
        query,
        session_id=session_id,
        use_latest_uploads_only=use_latest_uploads_only,
    )

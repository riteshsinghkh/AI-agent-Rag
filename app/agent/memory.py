"""
Session Memory
Implements in-memory session storage for conversation history.
"""

from typing import Dict, List
from app.config import settings


class SessionMemory:
    """In-memory storage for conversation sessions"""
    
    def __init__(self):
        """Initialize empty memory storage"""
        self._sessions: Dict[str, List[Dict]] = {}
    
    def add_message(self, session_id: str, role: str, content: str):
        """
        Add a message to session history
        
        Args:
            session_id: Session identifier
            role: Message role (user/assistant)
            content: Message content
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        
        self._sessions[session_id].append({
            "role": role,
            "content": content
        })
        
        # Limit history to MAX_HISTORY_MESSAGES
        max_messages = settings.MAX_HISTORY_MESSAGES
        if len(self._sessions[session_id]) > max_messages:
            self._sessions[session_id] = self._sessions[session_id][-max_messages:]
    
    def get_history(self, session_id: str) -> List[Dict]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of message dictionaries
        """
        return self._sessions.get(session_id, [])
    
    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self._sessions:
            del self._sessions[session_id]


# Global memory instance
memory = SessionMemory()

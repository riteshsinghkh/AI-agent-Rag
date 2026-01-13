"""
Hugging Face LLM Client
LLM provider using Hugging Face Inference Router (2026 Standard).
"""

import os
import logging
import requests
from typing import List, Dict
from dotenv import load_dotenv

from app.llm.base import LLMClient

load_dotenv()
logger = logging.getLogger(__name__)


class HuggingFaceClient(LLMClient):
    """Hugging Face Inference Router LLM provider (2026 OpenAI-compatible API)"""
    
    def __init__(self):
        """Initialize Hugging Face client with 2026 Router"""
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        # Default to a modern, reliable model with good free-tier support
        # Recommended: meta-llama/Llama-3.2-3B-Instruct, mistralai/Mistral-Nemo-Instruct-2407
        self.model = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
        
        # 2026 STANDARD: Global Unified Router URL
        # IMPORTANT: Do NOT put the model name in this URL!
        # The router identifies the model from the "model" field in the JSON payload
        self.router_url = "https://router.huggingface.co/v1/chat/completions"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized HuggingFace Unified Router (2026) for model: {self.model}")
        logger.info("Using model-agnostic endpoint (model specified in payload)")
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate chat completion using Hugging Face Router (2026 Standard).
        Uses OpenAI-compatible API format.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        # Check if API key is set
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY is required but not set. Please set it in your .env file.")
        
        # The payload format is now identical to OpenAI's
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            logger.info(f"Calling HuggingFace Unified Router for model: {self.model}")
            
            # Hit the generic endpoint (model-agnostic URL)
            response = requests.post(
                self.router_url, 
                headers=self.headers, 
                json=payload, 
                timeout=60  # 1 minute timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                # OpenAI-compatible response format
                generated_text = result['choices'][0]['message']['content'].strip()
                logger.info("Successfully received response from HuggingFace Unified Router")
                return generated_text
            
            else:
                # Handle errors
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', {}).get('message', response.text[:200])
                except Exception:
                    error_msg = response.text[:200]
                
                logger.error(f"HuggingFace Router Error ({response.status_code}): {error_msg}")
                raise Exception(f"HuggingFace Router failed ({response.status_code}): {error_msg}")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error calling HuggingFace Router: {e}")
            raise Exception(f"Failed to connect to HuggingFace Router: {e}") from e
        except Exception as e:
            logger.error(f"HuggingFace Router error: {e}")
            raise
    
    def get_provider_name(self) -> str:
        """Return provider name"""
        return f"Hugging Face ({self.model})"

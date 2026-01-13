"""
Azure OpenAI LLM Client
Primary LLM provider using Azure OpenAI Service.
"""

import os
import logging
from typing import List, Dict
from openai import AzureOpenAI
from dotenv import load_dotenv

from app.llm.base import LLMClient

load_dotenv()
logger = logging.getLogger(__name__)


class AzureOpenAIClient(LLMClient):
    """Azure OpenAI LLM provider"""
    
    def __init__(self):
        """Initialize Azure OpenAI client"""
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
        logger.info(f"Initialized Azure OpenAI client with deployment: {self.deployment_name}")
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate chat completion using Azure OpenAI.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            raise
    
    def get_provider_name(self) -> str:
        """Return provider name"""
        return "Azure OpenAI"

"""
LLM Client Factory
Creates the appropriate LLM client based on configuration.
"""

import os
import logging
from dotenv import load_dotenv

from app.llm.base import LLMClient
from app.llm.azure_client import AzureOpenAIClient
from app.llm.huggingface_client import HuggingFaceClient
from app.llm.local_model_client import LocalModelClient

load_dotenv()
logger = logging.getLogger(__name__)


def get_llm_client() -> LLMClient:
    """
    Factory function to get the appropriate LLM client.
    
    Selects provider based on LLM_PROVIDER environment variable:
    - "azure" → Azure OpenAI (primary)
    - "huggingface" → Hugging Face Inference API (fallback)
    - "local" → Local HuggingFace model (runs on your machine)
    
    Returns:
        LLMClient instance
    """
    provider = os.getenv("LLM_PROVIDER", "local").lower()
    
    if provider == "azure":
        logger.info("Using Azure OpenAI as LLM provider")
        return AzureOpenAIClient()
    elif provider == "huggingface":
        logger.info("Using Hugging Face as LLM provider")
        return HuggingFaceClient()
    elif provider == "local":
        logger.info("Using Local Model as LLM provider")
        model_name = os.getenv("LOCAL_MODEL_NAME", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
        return LocalModelClient(model_name=model_name)
    else:
        logger.warning(f"Unknown LLM provider '{provider}', defaulting to Local Model")
        model_name = os.getenv("LOCAL_MODEL_NAME", "HuggingFaceTB/SmolLM2-1.7B-Instruct")
        return LocalModelClient(model_name=model_name)

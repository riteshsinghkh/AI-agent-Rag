"""
Hugging Face LLM Client
LLM provider using Hugging Face Inference Router (2026 Standard).
"""

import os
import logging
import requests
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

from app.llm.base import LLMClient

load_dotenv()
logger = logging.getLogger(__name__)


class HuggingFaceRouterError(RuntimeError):
    """Raised when Hugging Face router requests fail."""


class HuggingFaceClient(LLMClient):
    """Hugging Face Inference Router LLM provider (2026 OpenAI-compatible API)"""
    
    def __init__(self):
        """Initialize Hugging Face client with 2026 Router"""
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        # Primary model can be overridden; fallback list is used when provider support differs by account.
        self.model = os.getenv("HUGGINGFACE_MODEL", "mistralai/Mistral-Nemo-Instruct-v1")
        self.fallback_models = self._parse_fallback_models(
            os.getenv(
                "HUGGINGFACE_FALLBACK_MODELS",
                "Qwen/Qwen2.5-7B-Instruct,HuggingFaceTB/SmolLM2-1.7B-Instruct",
            )
        )
        self.model_candidates = self._build_model_candidates(self.model, self.fallback_models)
        
        # 2026 STANDARD: Global Unified Router URL
        # IMPORTANT: Do NOT put the model name in this URL!
        # The router identifies the model from the "model" field in the JSON payload
        self.router_url = "https://router.huggingface.co/v1/chat/completions"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info("Initialized HuggingFace Unified Router (2026) for model: %s", self.model)
        logger.info("HuggingFace fallback model candidates: %s", self.model_candidates)
        logger.info("Using model-agnostic endpoint (model specified in payload)")

    @staticmethod
    def _parse_fallback_models(raw_value: str) -> List[str]:
        """Parse comma-separated fallback models from environment."""
        return [item.strip() for item in raw_value.split(",") if item.strip()]

    @staticmethod
    def _build_model_candidates(primary: str, fallbacks: List[str]) -> List[str]:
        """Build ordered de-duplicated model candidate list."""
        ordered = [primary] + list(fallbacks)
        result: List[str] = []
        seen = set()
        for model_name in ordered:
            normalized = model_name.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    @staticmethod
    def _is_retryable_model_error(status_code: int, error_msg: str) -> bool:
        """Return True when request should retry with next model candidate."""
        if status_code not in {400, 404}:
            return False

        lowered = (error_msg or "").lower()
        retry_markers = [
            "not supported by any provider",
            "requested model",
            "model is not supported",
            "model not found",
            "unknown model",
        ]
        return any(marker in lowered for marker in retry_markers)

    def _router_request(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """Execute one router request and return (text, error_message, status_code)."""
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        try:
            response = requests.post(
                self.router_url,
                headers=self.headers,
                json=payload,
                timeout=60,
            )
        except requests.exceptions.RequestException as e:
            logger.error("Network error calling HuggingFace Router: %s", e)
            raise HuggingFaceRouterError(f"Failed to connect to HuggingFace Router: {e}") from e

        if response.status_code == 200:
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"].strip()
            return generated_text, None, 200

        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text[:300])
        except ValueError:
            error_msg = response.text[:300]

        return None, error_msg, response.status_code
    
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

        errors: List[str] = []
        for model_name in self.model_candidates:
            logger.info("Calling HuggingFace Unified Router for model: %s", model_name)
            text, error_msg, status_code = self._router_request(
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if text is not None:
                if self.model != model_name:
                    logger.warning("Switching active HuggingFace model to fallback: %s", model_name)
                    self.model = model_name
                logger.info("Successfully received response from HuggingFace Unified Router")
                return text

            assert error_msg is not None
            assert status_code is not None

            logger.error("HuggingFace Router Error (%s) for %s: %s", status_code, model_name, error_msg)
            errors.append(f"{model_name} -> ({status_code}) {error_msg}")

            if not self._is_retryable_model_error(status_code, error_msg):
                break

        joined_errors = " | ".join(errors)
        raise HuggingFaceRouterError(
            f"HuggingFace Router failed after model fallback attempts: {joined_errors}"
        )
    
    def get_provider_name(self) -> str:
        """Return provider name"""
        return f"Hugging Face ({self.model})"

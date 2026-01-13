"""
Local Model Client
LLM provider using locally loaded HuggingFace models with transformers.
"""

import logging
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.llm.base import LLMClient

logger = logging.getLogger(__name__)


class LocalModelClient(LLMClient):
    """Local HuggingFace Model LLM provider"""
    
    def __init__(self, model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"):
        """
        Initialize Local Model client
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading local model: {model_name} on device: {self.device}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            logger.info(f"Successfully loaded local model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load local model {model_name}: {e}")
            raise
    
    def _format_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert chat messages to a properly formatted prompt.
        
        Args:
            messages: List of message dicts
            
        Returns:
            Formatted prompt string
        """
        # Use the tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Failed to use chat template: {e}, using fallback formatting")
        
        # Fallback formatting for SmolLM and similar models
        prompt_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
        
        # Add assistant prompt for generation
        prompt_parts.append("<|im_start|>assistant\n")
        
        return "".join(prompt_parts)
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate chat completion using local model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        try:
            # Format messages into a prompt
            prompt = self._format_messages_to_prompt(messages)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove the prompt)
            generated_text = full_response[len(prompt):].strip()
            
            # Clean up any remaining special tokens
            stop_strings = ["<|im_end|>", "<|endoftext|>", "</s>"]
            for stop_str in stop_strings:
                if stop_str in generated_text:
                    generated_text = generated_text.split(stop_str)[0].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Local model generation error: {e}")
            raise
    
    def get_provider_name(self) -> str:
        """Return provider name"""
        return f"Local Model ({self.model_name})"

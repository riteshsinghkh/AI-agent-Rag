"""
Configuration Management
Loads environment variables and manages application configuration.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables"""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    # RAG Configuration
    CHUNK_SIZE: int = 400  # tokens
    CHUNK_OVERLAP: int = 50  # tokens
    TOP_K: int = 3  # number of documents to retrieve
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))

    # Seed docs
    SEED_DOCS: bool = os.getenv("SEED_DOCS", "false").strip().lower() in {"1", "true", "yes"}

    # Uploads
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "data/uploads")
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    
    # Session Memory Configuration
    MAX_HISTORY_MESSAGES: int = 10


# Create settings instance
settings = Settings()

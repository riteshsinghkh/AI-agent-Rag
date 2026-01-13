"""
FastAPI Application Entry Point
This is the main file that initializes the FastAPI app and includes routers.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.api.ask import router as ask_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown.
    Initializes RAG system on startup.
    """
    # Startup
    logger.info("Starting AI Agent RAG API...")
    
    try:
        # Initialize RAG system (build/load FAISS index)
        from app.rag.retriever import initialize_rag
        logger.info("Initializing RAG system...")
        initialize_rag()
        logger.info("RAG system initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        logger.warning("API will start but RAG may not work properly")
    
    yield  # App runs here
    
    # Shutdown
    logger.info("Shutting down AI Agent RAG API...")


# Create FastAPI app
app = FastAPI(
    title="AI Agent RAG API",
    description="""
## AI Agent with RAG Capabilities

This API provides an intelligent assistant that can:
- Answer general questions directly
- Search company documents for policy-related queries
- Maintain conversation context within sessions

### Features:
- **RAG (Retrieval-Augmented Generation)**: Uses FAISS vector store for document search
- **Session Memory**: Maintains conversation history per session
- **Azure OpenAI**: Powered by GPT-4o-mini and text-embedding-3-small

### Endpoints:
- `POST /ask` - Ask the AI agent a question
- `GET /health` - Health check for monitoring
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://wonderful-sky-0b2ff020f.1.azurestaticapps.net"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(ask_router, tags=["Agent"])

# Mount static files for UI
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for Azure App Service.
    
    Returns:
        dict: Status indicator
    """
    return {"status": "healthy"}


@app.get("/", tags=["Root"])
async def root():
    """
    Serve the chat UI.
    
    Returns:
        HTML: Chat interface
    """
    static_path = Path(__file__).parent / "static" / "index.html"
    if static_path.exists():
        return FileResponse(static_path, media_type="text/html")
    
    return {
        "message": "Welcome to AI Agent RAG API",
        "docs": "/docs",
        "health": "/health",
        "ask": "POST /ask"
    }

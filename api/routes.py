import os
import logging
from typing import Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Depends
from omegaconf import OmegaConf
from models.reponse import info_response, ragit_response
from models.request import ragit_request
from rag_service.rag_pipeline import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGConfig:
    """Configuration management for RAG service"""
    
    def __init__(self):
        status = load_dotenv()
        if not status:
            logger.warning("No .env file found or failed to load")
        
        self.embedding_model = self._get_required_env("EMBEDDING_MODEL")
        self.generation_model = self._get_required_env("GENERATION_MODEL")
        self.vector_store_path = self._get_required_env("VECTOR_STORE_PATH")
        self.vector_store_index = self._get_required_env("VECTOR_STORE_INDEX")
    
    def _get_required_env(self, key: str) -> str:
        """Get required environment variable with validation"""
        value = os.environ.get(key, "").strip()
        if not value:
            raise ValueError(f"Required environment variable {key} is not set or empty")
        return value
    
    def to_omega_conf(self) -> OmegaConf:
        """Convert config to OmegaConf object"""
        return OmegaConf.create({
            "embedding_model": self.embedding_model,
            "output_vector_store_path": self.vector_store_path,
            "output_vector_store_index_name": self.vector_store_index
        })


class RAGService:
    """RAG service with lazy initialization and error handling"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self._vector_store_retriever = None
        self._mq_retriever = None
        self._chain = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize RAG components lazily"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing RAG service components...")
            args = self.config.to_omega_conf()
            
            # Initialize components with error handling
            self._vector_store_retriever = set_up_vector_store_retriver(args)
            self._mq_retriever = set_up_multi_query_retriever(
                self.config.generation_model, 
                self._vector_store_retriever
            )
            self._chain = create_chain(self.config.generation_model)
            
            self._initialized = True
            logger.info("RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"RAG service initialization failed: {str(e)}"
            )
    
    async def query(self, query: str) -> dict:
        """Execute RAG query with error handling"""
        await self.initialize()
        
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        try:
            result = rag_query(query, self._mq_retriever, self._chain)
            
            # Validate result structure
            if not hasattr(result, 'id') or not hasattr(result, 'content'):
                raise ValueError("Invalid result structure from rag_query")
            
            return {
                'id': result.id,
                'content': result.content,
                'model': result.response_metadata.get('model', 'unknown') if hasattr(result, 'response_metadata') else 'unknown'
            }
            
        except Exception as e:
            logger.error(f"RAG query failed: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"RAG query processing failed: {str(e)}"
            )
    
    async def cleanup(self):
        """Cleanup resources if needed"""
        logger.info("RAG service cleanup completed")

# Global instances (initialized lazily)
try:
    rag_config = RAGConfig()
    rag_service = RAGService(rag_config)
except Exception as e:
    logger.error(f"Failed to create RAG configuration: {str(e)}")
    raise

# Dependency injection
async def get_rag_service() -> RAGService:
    """Dependency injection for RAG service"""
    return rag_service

# Router setup
rag_router = APIRouter()

@rag_router.get(path='/info', response_model=info_response)
async def info(service: RAGService = Depends(get_rag_service)):
    """
    Return the current configuration of the RAG pipeline
    """
    try:
        return info_response(
            embedding_model=service.config.embedding_model,
            generation_model=service.config.generation_model,
            vector_store_path=service.config.vector_store_path,
            vector_store_index=service.config.vector_store_index
        )
    except Exception as e:
        logger.error(f"Failed to get config info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve configuration")


@rag_router.post(path='/ragit', response_model=ragit_response)
async def rag_it(
    request: ragit_request, 
    service: RAGService = Depends(get_rag_service)
):
    """
    Process RAG query and return results
    """
    try:
        result = await service.query(request.query)
        
        return ragit_response(
            id=result['id'],
            user_query=request.query,
            rag_result=result['content'],
            model_used=result['model']
        )
        
    except HTTPException:
        raise  
    except Exception as e:
        logger.error(f"Unexpected error in ragit endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@asynccontextmanager
async def lifespan(app):
    """Manage application lifespan"""
    logger.info("Starting RAG application")
    yield
    logger.info("Shutting down RAG application")
    await rag_service.cleanup()
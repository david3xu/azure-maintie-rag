"""
Simple FastAPI Main - CODING_STANDARDS Compliant
Clean API server without over-engineering.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.endpoints.search import router as search_router

# Create FastAPI app
app = FastAPI(
    title="Azure Universal RAG API",
    description="Simple API for universal search and document processing",
    version="1.0.0",
)

# Simple CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search_router)


@app.get("/")
async def root():
    """Simple root endpoint"""
    return {
        "message": "Azure Universal RAG API - FUNC! Real Azure Services Only",
        "version": "1.0.0", 
        "endpoints": ["/api/v1/search", "/api/v1/extract", "/api/v1/health", "/api/v1/stream/workflow/{query_id}"],
        "data_source": "REAL Azure data from data/raw/azure-ai-services-language-service_output/",
        "principles": "NO fake data, NO simulation, QUICK FAIL on errors"
    }


@app.get("/health")
async def health_check():
    """Simple health check"""
    return {"status": "healthy"}

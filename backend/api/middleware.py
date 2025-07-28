"""
API Middleware Configuration
Extracted from main.py for clean separation of concerns
"""

import time
import logging
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from config.settings import settings

logger = logging.getLogger(__name__)


def configure_middleware(app):
    """Configure all middleware for the FastAPI application"""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5174", "http://localhost:5175"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add trusted host middleware for security (if configured)
    if hasattr(settings, 'trusted_hosts_list') and settings.trusted_hosts_list:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.trusted_hosts_list
        )
    
    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests"""
        start_time = time.time()
        
        # Skip logging for health checks and static files
        if request.url.path in ["/health", "/docs", "/openapi.json", "/favicon.ico"]:
            return await call_next(request)
        
        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"Response: {request.method} {request.url.path} "
            f"Status: {response.status_code} Duration: {process_time:.3f}s"
        )
        
        # Add custom headers
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    logger.info("✅ Middleware configured successfully")
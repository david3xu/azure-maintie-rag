"""
Progress Streaming Module
Handles real-time progress updates for long-running operations
Extracted from workflow_stream.py for focused functionality
"""

import asyncio
import json
from typing import AsyncGenerator, Dict, Any
from datetime import datetime
from fastapi import Response
from fastapi.responses import StreamingResponse

import logging
logger = logging.getLogger(__name__)


class ProgressStreamer:
    """Handles streaming of progress updates to clients"""
    
    def __init__(self):
        self.active_streams = {}
    
    async def stream_progress(self, operation_id: str) -> AsyncGenerator[str, None]:
        """Stream progress updates for a specific operation"""
        try:
            while operation_id in self.active_streams:
                progress_data = self.active_streams.get(operation_id, {})
                
                # Format as Server-Sent Event
                event_data = {
                    "id": operation_id,
                    "timestamp": datetime.now().isoformat(),
                    "progress": progress_data.get("progress", 0),
                    "status": progress_data.get("status", "in_progress"),
                    "message": progress_data.get("message", "Processing...")
                }
                
                yield f"data: {json.dumps(event_data)}\n\n"
                
                # Check if operation is complete
                if progress_data.get("status") in ["completed", "failed"]:
                    break
                
                await asyncio.sleep(0.5)  # Update frequency
                
        except asyncio.CancelledError:
            logger.info(f"Progress stream cancelled for operation {operation_id}")
            raise
        finally:
            # Clean up
            self.active_streams.pop(operation_id, None)
    
    def update_progress(self, operation_id: str, progress: float, status: str, message: str):
        """Update progress for an operation"""
        self.active_streams[operation_id] = {
            "progress": progress,
            "status": status,
            "message": message,
            "updated_at": datetime.now().isoformat()
        }
    
    def create_progress_response(self, operation_id: str) -> StreamingResponse:
        """Create a streaming response for progress updates"""
        return StreamingResponse(
            self.stream_progress(operation_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable Nginx buffering
            }
        )


# Global progress streamer instance
progress_streamer = ProgressStreamer()
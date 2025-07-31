"""
Progress Streaming Module
========================

Manages progress streaming for long-running operations
"""

import asyncio
import json
import time
from typing import Dict, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum


class ProgressStatus(Enum):
    """Progress status enumeration"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProgressUpdate:
    """Progress update data structure"""
    operation_id: str
    progress: float
    status: ProgressStatus
    message: str
    timestamp: float
    details: Dict[str, Any] = None


class ProgressStreamer:
    """Manages progress streaming for operations"""
    
    def __init__(self):
        self.active_streams: Dict[str, Dict[str, Any]] = {}
    
    def update_progress(self, operation_id: str, progress: float, status: str, message: str, details: Dict = None):
        """Update progress for an operation"""
        self.active_streams[operation_id] = {
            'progress': progress,
            'status': status,
            'message': message,
            'timestamp': time.time(),
            'details': details or {}
        }
    
    async def stream_progress(self, operation_id: str) -> AsyncGenerator[str, None]:
        """Stream progress updates for an operation"""
        if operation_id not in self.active_streams:
            yield "data: {\"error\": \"Operation not found\"}\n\n"
            return
        
        operation_data = self.active_streams[operation_id]
        
        # Format as Server-Sent Events
        data = {
            "operation_id": operation_id,
            "progress": operation_data['progress'],
            "status": operation_data['status'],
            "message": operation_data['message'],
            "timestamp": operation_data['timestamp']
        }
        
        yield f"data: {json.dumps(data)}\n\n"
    
    def complete_operation(self, operation_id: str, final_message: str = "Operation completed"):
        """Mark an operation as completed"""
        if operation_id in self.active_streams:
            self.active_streams[operation_id].update({
                'progress': 100.0,
                'status': 'completed',
                'message': final_message,
                'timestamp': time.time()
            })
    
    def fail_operation(self, operation_id: str, error_message: str):
        """Mark an operation as failed"""
        if operation_id in self.active_streams:
            self.active_streams[operation_id].update({
                'status': 'failed', 
                'message': error_message,
                'timestamp': time.time()
            })
    
    def cleanup_completed_operations(self, max_age: int = 3600):
        """Clean up old completed operations"""
        current_time = time.time()
        to_remove = []
        
        for operation_id, operation_data in self.active_streams.items():
            if (operation_data['status'] in ['completed', 'failed'] and 
                current_time - operation_data['timestamp'] > max_age):
                to_remove.append(operation_id)
        
        for operation_id in to_remove:
            del self.active_streams[operation_id]
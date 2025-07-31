"""
Streaming Models
Models for real-time streaming responses
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Literal
from datetime import datetime
from enum import Enum


class StreamEventType(str, Enum):
    """Types of streaming events"""
    PROGRESS = "progress"
    DATA = "data"
    ERROR = "error"
    COMPLETE = "complete"
    HEARTBEAT = "heartbeat"


class StreamEvent(BaseModel):
    """Base streaming event model"""
    id: str = Field(description="Unique event identifier")
    type: StreamEventType = Field(description="Type of streaming event")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")
    sequence: int = Field(description="Event sequence number")


class ProgressEvent(StreamEvent):
    """Progress update event"""
    type: StreamEventType = StreamEventType.PROGRESS
    progress: float = Field(description="Progress percentage", ge=0.0, le=100.0)
    message: str = Field(description="Progress message")
    current_step: Optional[str] = Field(None, description="Current step name")
    total_steps: Optional[int] = Field(None, description="Total number of steps")


class DataEvent(StreamEvent):
    """Data chunk event"""
    type: StreamEventType = StreamEventType.DATA
    data: Dict[str, Any] = Field(description="Data payload")
    chunk_index: Optional[int] = Field(None, description="Chunk sequence number")
    is_final: bool = Field(False, description="Whether this is the final chunk")


class ErrorEvent(StreamEvent):
    """Error event"""
    type: StreamEventType = StreamEventType.ERROR
    error_code: str = Field(description="Error code")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    recoverable: bool = Field(False, description="Whether the error is recoverable")


class CompleteEvent(StreamEvent):
    """Stream completion event"""
    type: StreamEventType = StreamEventType.COMPLETE
    summary: Dict[str, Any] = Field(description="Summary of the completed operation")
    duration_ms: int = Field(description="Total operation duration in milliseconds")
    success: bool = Field(description="Whether the operation completed successfully")


class HeartbeatEvent(StreamEvent):
    """Heartbeat event to keep connection alive"""
    type: StreamEventType = StreamEventType.HEARTBEAT
    message: str = Field(default="alive", description="Heartbeat message")


class StreamingRequest(BaseModel):
    """Base model for requests that support streaming"""
    stream: bool = Field(False, description="Whether to stream the response")
    stream_options: Optional[Dict[str, Any]] = Field(None, description="Streaming configuration options")


class StreamOptions(BaseModel):
    """Configuration options for streaming"""
    chunk_size: int = Field(1024, description="Size of data chunks")
    heartbeat_interval: int = Field(30, description="Heartbeat interval in seconds")
    include_metadata: bool = Field(True, description="Include metadata in stream events")
    compression: Optional[str] = Field(None, description="Compression type (gzip, deflate)")
    format: Literal["json", "ndjson", "sse"] = Field("sse", description="Stream format")
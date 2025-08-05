"""
Bidirectional Communication Hub Between Dual Graphs

This module implements the core communication system that enables intelligent
coordination between Config-Extraction and Search graphs.
"""

import asyncio
import json
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Types of messages that can be exchanged between graphs"""
    CONFIG_REQUEST = "config_request"
    CONFIG_OFFER = "config_offer"
    PERFORMANCE_FEEDBACK = "performance_feedback"
    OPTIMIZATION_SUGGESTION = "optimization_suggestion"
    STATUS_UPDATE = "status_update"
    HANDSHAKE_INIT = "handshake_init"
    HANDSHAKE_ACK = "handshake_ack"


class GraphMessage(BaseModel):
    """Message format for graph-to-graph communication"""
    message_id: str = Field(default_factory=lambda: str(uuid4()))
    source_graph: str
    target_graph: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    requires_response: bool = False
    correlation_id: Optional[str] = None


class GraphStatus(BaseModel):
    """Status information about a graph's current state"""
    graph_id: str
    status: str  # "ready", "processing", "waiting", "error"
    domain: Optional[str] = None
    capabilities: List[str] = []
    config_updated: bool = False
    performance_metrics: Dict[str, float] = {}
    last_updated: datetime = Field(default_factory=datetime.now)


class GraphComm:
    """Bidirectional communication hub between dual graphs"""

    def __init__(self):
        self._message_queues: Dict[str, asyncio.Queue] = {}
        self._status_registry: Dict[str, GraphStatus] = {}
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    async def initialize_graph(self, graph_id: str) -> None:
        """Initialize communication for a graph"""
        if graph_id not in self._message_queues:
            self._message_queues[graph_id] = asyncio.Queue()
            self._locks[graph_id] = asyncio.Lock()

    async def initiate_handshake(
        self,
        source_graph: str,
        target_graph: str,
        intent: str
    ) -> bool:
        """Initiate handshake protocol between two graphs"""
        await self.initialize_graph(source_graph)
        await self.initialize_graph(target_graph)

        handshake_message = GraphMessage(
            source_graph=source_graph,
            target_graph=target_graph,
            message_type=MessageType.HANDSHAKE_INIT,
            payload={"intent": intent},
            requires_response=True
        )

        try:
            response = await self._send_and_wait_for_response(
                handshake_message,
                timeout=5.0
            )
            return response.message_type == MessageType.HANDSHAKE_ACK
        except asyncio.TimeoutError:
            return False

    async def send_message(self, message: GraphMessage) -> Optional[GraphMessage]:
        """Send a message to another graph"""
        await self.initialize_graph(message.target_graph)

        async with self._locks[message.target_graph]:
            await self._message_queues[message.target_graph].put(message)

        if message.requires_response:
            return await self._wait_for_response(message.message_id)
        return None

    async def listen_for_requests(self, graph_id: str) -> AsyncIterator[GraphMessage]:
        """Listen for incoming messages for a specific graph"""
        await self.initialize_graph(graph_id)

        while True:
            try:
                message = await asyncio.wait_for(
                    self._message_queues[graph_id].get(),
                    timeout=1.0
                )
                yield message
            except asyncio.TimeoutError:
                continue

    async def send_response(
        self,
        original_message: GraphMessage,
        response_payload: Dict[str, Any]
    ) -> None:
        """Send a response to an original message"""
        response = GraphMessage(
            source_graph=original_message.target_graph,
            target_graph=original_message.source_graph,
            message_type=MessageType.CONFIG_OFFER,  # Default response type
            payload=response_payload,
            correlation_id=original_message.message_id
        )

        if original_message.message_id in self._pending_responses:
            self._pending_responses[original_message.message_id].set_result(response)

    async def broadcast_status_update(
        self,
        graph_id: str,
        status: GraphStatus
    ) -> None:
        """Broadcast status update to all other graphs"""
        self._status_registry[graph_id] = status

        # Send status update to all other graphs
        for target_graph in self._message_queues.keys():
            if target_graph != graph_id:
                status_message = GraphMessage(
                    source_graph=graph_id,
                    target_graph=target_graph,
                    message_type=MessageType.STATUS_UPDATE,
                    payload=status.model_dump()
                )
                await self.send_message(status_message)

    def get_graph_status(self, graph_id: str) -> Optional[GraphStatus]:
        """Get current status of a graph"""
        return self._status_registry.get(graph_id)

    def get_all_statuses(self) -> Dict[str, GraphStatus]:
        """Get status of all registered graphs"""
        return self._status_registry.copy()

    async def _send_and_wait_for_response(
        self,
        message: GraphMessage,
        timeout: float = 10.0
    ) -> GraphMessage:
        """Send message and wait for response"""
        future = asyncio.Future()
        self._pending_responses[message.message_id] = future

        await self.send_message(message)

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        finally:
            self._pending_responses.pop(message.message_id, None)

    async def _wait_for_response(
        self,
        message_id: str,
        timeout: float = 10.0
    ) -> Optional[GraphMessage]:
        """Wait for a response to a specific message"""
        if message_id not in self._pending_responses:
            return None

        try:
            return await asyncio.wait_for(
                self._pending_responses[message_id],
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

    async def cleanup(self) -> None:
        """Clean up resources and pending operations"""
        # Cancel all pending responses
        for future in self._pending_responses.values():
            if not future.done():
                future.cancel()

        # Clear all queues
        for queue in self._message_queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        self._message_queues.clear()
        self._status_registry.clear()
        self._pending_responses.clear()
        self._locks.clear()


# Global instance for graph communication
graph_comm = GraphComm()

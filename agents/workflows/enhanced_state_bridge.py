"""
Enhanced State Bridge for Inter-Graph Communication

This module implements an enhanced state management system that enables clean
communication between the dual workflow graphs while maintaining proper
separation of concerns and data integrity.

Key Features:
- Type-safe state transfer between Config-Extraction and Search workflows
- Configuration dependency tracking and freshness validation
- Event-driven state synchronization
- State versioning and rollback capabilities
- Performance monitoring and optimization
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field

# Import consolidated data models
from agents.core.data_models import StateTransferPacket, GraphConnectionInfo
from enum import Enum
import json
from pathlib import Path

from agents.core.data_models import (
    WorkflowState,
    NodeState,
    WorkflowExecutionState,
    NodeExecutionResult,
)
from agents.workflows.state_persistence import WorkflowStateManager
from agents.core.constants import CacheConstants, WorkflowConstants

logger = logging.getLogger(__name__)


class StateTransferType(str, Enum):
    """Types of state transfers between workflow graphs"""

    CONFIG_GENERATION = "config_generation"
    DOMAIN_ANALYSIS = "domain_analysis"
    PATTERN_LEARNING = "pattern_learning"
    PERFORMANCE_METRICS = "performance_metrics"
    ERROR_CONTEXT = "error_context"


# StateTransferPacket and GraphConnectionInfo now imported from agents.core.data_models


class EnhancedStateBridge:
    """
    Enhanced state bridge for managing communication between dual workflow graphs.

    Provides:
    - Type-safe state transfers with integrity checking
    - Dependency tracking between graph outputs
    - Event-driven notifications for state changes
    - Performance monitoring and optimization
    """

    def __init__(self):
        self.state_manager = WorkflowStateManager()
        self.transfer_queue: Dict[str, StateTransferPacket] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.connection_info: Dict[str, GraphConnectionInfo] = {}
        self.event_listeners: Dict[str, List[callable]] = {}

        # Initialize graph connections
        self._initialize_graph_connections()

    def _initialize_graph_connections(self):
        """Initialize connections between workflow graphs"""

        # Config-Extraction → Search workflow connection
        self.connection_info["config_to_search"] = GraphConnectionInfo(
            source_graph="config_extraction",
            target_graph="search",
            connection_type="configuration_dependency",
            data_flow_direction="unidirectional",
        )

        # Search → Config-Extraction feedback connection (for learning)
        self.connection_info["search_to_config"] = GraphConnectionInfo(
            source_graph="search",
            target_graph="config_extraction",
            connection_type="performance_feedback",
            data_flow_direction="feedback",
        )

    async def transfer_state(
        self,
        source_workflow: str,
        target_workflow: str,
        transfer_type: StateTransferType,
        payload: Dict[str, Any],
        dependencies: List[str] = None,
    ) -> str:
        """
        Transfer state between workflow graphs with integrity and dependency tracking.

        Args:
            source_workflow: Source workflow identifier
            target_workflow: Target workflow identifier
            transfer_type: Type of state transfer
            payload: Data to transfer
            dependencies: List of dependency transfer IDs

        Returns:
            Transfer ID for tracking
        """

        transfer_id = (
            f"{source_workflow}_to_{target_workflow}_{int(datetime.now().timestamp())}"
        )

        try:
            # Create transfer packet
            packet = StateTransferPacket(
                transfer_id=transfer_id,
                transfer_type=transfer_type,
                source_workflow=source_workflow,
                target_workflow=target_workflow,
                payload=payload,
                dependencies=dependencies or [],
            )

            # Validate dependencies
            if not await self._validate_dependencies(packet.dependencies):
                raise Exception(
                    f"Transfer dependencies not satisfied: {packet.dependencies}"
                )

            # Add to transfer queue
            self.transfer_queue[transfer_id] = packet

            # Update dependency graph
            if transfer_type == StateTransferType.CONFIG_GENERATION:
                self.dependency_graph[target_workflow] = self.dependency_graph.get(
                    target_workflow, []
                ) + [transfer_id]

            # Persist transfer for recovery
            await self._persist_transfer(packet)

            # Update connection statistics
            connection_key = f"{source_workflow}_to_{target_workflow}"
            if connection_key in self.connection_info:
                conn_info = self.connection_info[connection_key]
                conn_info.total_transfers += 1
                conn_info.last_transfer = datetime.now(timezone.utc)

            # Trigger event listeners
            await self._notify_listeners(
                "state_transferred",
                {
                    "transfer_id": transfer_id,
                    "transfer_type": transfer_type.value,
                    "source": source_workflow,
                    "target": target_workflow,
                },
            )

            logger.info(
                f"✅ State transferred: {source_workflow} → {target_workflow} ({transfer_type.value})"
            )
            return transfer_id

        except Exception as e:
            logger.error(
                f"❌ State transfer failed: {source_workflow} → {target_workflow}: {e}"
            )

            # Update failure statistics
            connection_key = f"{source_workflow}_to_{target_workflow}"
            if connection_key in self.connection_info:
                self.connection_info[connection_key].failed_transfers += 1

            raise

    async def get_state(
        self,
        target_workflow: str,
        transfer_type: StateTransferType = None,
        max_age_hours: int = 24,
    ) -> Optional[StateTransferPacket]:
        """
        Get the latest state transfer for a target workflow.

        Args:
            target_workflow: Target workflow to get state for
            transfer_type: Optional filter by transfer type
            max_age_hours: Maximum age of transfer to consider

        Returns:
            Latest matching state transfer packet
        """

        cutoff_time = datetime.now(timezone.utc).replace(
            hour=datetime.now(timezone.utc).hour - max_age_hours
        )

        matching_transfers = []

        for packet in self.transfer_queue.values():
            # Filter by target workflow
            if packet.target_workflow != target_workflow:
                continue

            # Filter by transfer type if specified
            if transfer_type and packet.transfer_type != transfer_type:
                continue

            # Filter by age
            if packet.timestamp < cutoff_time:
                continue

            # Check if expired
            if packet.is_expired():
                continue

            # Validate integrity
            if not packet.validate_integrity():
                logger.warning(
                    f"⚠️  State packet integrity check failed: {packet.transfer_id}"
                )
                continue

            matching_transfers.append(packet)

        if not matching_transfers:
            return None

        # Return the most recent transfer
        latest_transfer = max(matching_transfers, key=lambda p: p.timestamp)

        logger.debug(
            f"📋 Retrieved state for {target_workflow}: {latest_transfer.transfer_id}"
        )
        return latest_transfer

    async def wait_for_dependency(
        self,
        target_workflow: str,
        dependency_type: StateTransferType,
        timeout_seconds: int = 300,
    ) -> Optional[StateTransferPacket]:
        """
        Wait for a specific dependency to become available.

        Args:
            target_workflow: Workflow waiting for dependency
            dependency_type: Type of dependency to wait for
            timeout_seconds: Maximum time to wait

        Returns:
            State packet when dependency is satisfied
        """

        logger.info(
            f"⏳ Waiting for dependency: {target_workflow} needs {dependency_type.value}"
        )

        start_time = datetime.now()
        timeout = datetime.now().replace(second=datetime.now().second + timeout_seconds)

        while datetime.now() < timeout:
            # Check if dependency is available
            dependency_state = await self.get_state(target_workflow, dependency_type)

            if dependency_state:
                logger.info(
                    f"✅ Dependency satisfied: {target_workflow} got {dependency_type.value}"
                )
                return dependency_state

            # Wait before checking again
            await asyncio.sleep(1.0)

        logger.error(
            f"⏰ Dependency timeout: {target_workflow} waiting for {dependency_type.value}"
        )
        return None

    async def register_event_listener(self, event_type: str, callback: callable):
        """Register callback for state transfer events"""

        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []

        self.event_listeners[event_type].append(callback)
        logger.debug(f"📡 Registered listener for {event_type}")

    async def get_dependency_chain(self, workflow: str) -> List[str]:
        """Get the complete dependency chain for a workflow"""

        return self.dependency_graph.get(workflow, [])

    async def cleanup_expired_transfers(self):
        """Clean up expired state transfers"""

        expired_transfers = [
            transfer_id
            for transfer_id, packet in self.transfer_queue.items()
            if packet.is_expired()
        ]

        for transfer_id in expired_transfers:
            del self.transfer_queue[transfer_id]

        if expired_transfers:
            logger.info(
                f"🗑️  Cleaned up {len(expired_transfers)} expired state transfers"
            )

    async def get_bridge_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the state bridge"""

        # Calculate statistics
        total_transfers = sum(
            conn.total_transfers for conn in self.connection_info.values()
        )
        total_failures = sum(
            conn.failed_transfers for conn in self.connection_info.values()
        )

        overall_success_rate = (
            (total_transfers - total_failures) / total_transfers
            if total_transfers > 0
            else 1.0
        )

        return {
            "active_transfers": len(self.transfer_queue),
            "total_transfers": total_transfers,
            "total_failures": total_failures,
            "overall_success_rate": overall_success_rate,
            "graph_connections": {
                key: {
                    "source": conn.source_graph,
                    "target": conn.target_graph,
                    "success_rate": conn.success_rate,
                    "last_transfer": (
                        conn.last_transfer.isoformat() if conn.last_transfer else None
                    ),
                }
                for key, conn in self.connection_info.items()
            },
            "dependency_chains": self.dependency_graph,
            "event_listeners": {
                event: len(listeners)
                for event, listeners in self.event_listeners.items()
            },
        }

    async def _validate_dependencies(self, dependency_ids: List[str]) -> bool:
        """Validate that all dependencies are satisfied"""

        for dep_id in dependency_ids:
            if dep_id not in self.transfer_queue:
                logger.warning(f"⚠️  Dependency not found: {dep_id}")
                return False

            packet = self.transfer_queue[dep_id]
            if packet.is_expired():
                logger.warning(f"⚠️  Dependency expired: {dep_id}")
                return False

            if not packet.validate_integrity():
                logger.warning(f"⚠️  Dependency integrity check failed: {dep_id}")
                return False

        return True

    async def _persist_transfer(self, packet: StateTransferPacket):
        """Persist transfer packet to storage for recovery"""

        await self.state_manager.save_workflow_state(
            f"state_transfer_{packet.transfer_id}",
            WorkflowState.COMPLETED,
            {
                "transfer_data": {
                    "transfer_id": packet.transfer_id,
                    "transfer_type": packet.transfer_type.value,
                    "source_workflow": packet.source_workflow,
                    "target_workflow": packet.target_workflow,
                    "payload": packet.payload,
                    "timestamp": packet.timestamp.isoformat(),
                    "expiry": packet.expiry.isoformat(),
                    "dependencies": packet.dependencies,
                    "version": packet.version,
                    "checksum": packet.checksum,
                }
            },
            workflow_type="state_transfer",
        )

    async def _notify_listeners(self, event_type: str, event_data: Dict[str, Any]):
        """Notify registered event listeners"""

        if event_type in self.event_listeners:
            for callback in self.event_listeners[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event_data)
                    else:
                        callback(event_data)
                except Exception as e:
                    logger.error(f"❌ Event listener error for {event_type}: {e}")


# Global state bridge instance
enhanced_state_bridge = EnhancedStateBridge()

# Export main components
__all__ = [
    "EnhancedStateBridge",
    "StateTransferPacket",
    "StateTransferType",
    "GraphConnectionInfo",
    "enhanced_state_bridge",
]

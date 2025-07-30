#!/usr/bin/env python3
"""
Streaming Monitor - Real-time Pipeline Progress Events
Provides streaming progress events to frontend progressive UI

This script implements the real-time streaming features mentioned in the README:
- Monitors pipeline execution across all stages
- Streams progress events for frontend consumption
- Provides WebSocket-based real-time updates
- Tracks performance metrics and bottlenecks
"""

import sys
import asyncio
import argparse
import json
import websockets
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Callable
import logging
from datetime import datetime
import uuid
import threading
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.infrastructure_service import InfrastructureService

logger = logging.getLogger(__name__)

@dataclass
class StreamingEvent:
    """Streaming event data structure"""
    event_id: str
    timestamp: str
    event_type: str  # pipeline_start, stage_start, stage_progress, stage_complete, pipeline_complete, error
    stage: str
    pipeline_id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]

class StreamingMonitor:
    """Real-time Pipeline Progress Monitor"""
    
    def __init__(self):
        self.infrastructure = InfrastructureService()
        self.active_pipelines: Dict[str, Dict[str, Any]] = {}
        self.event_subscribers: Set[websockets.WebSocketServerProtocol] = set()
        self.event_history: List[StreamingEvent] = []
        self.max_history_size = 1000
        
    async def register_pipeline(
        self,
        pipeline_id: str,
        pipeline_type: str,
        domain: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Register a new pipeline for monitoring
        
        Args:
            pipeline_id: Unique pipeline identifier
            pipeline_type: Type of pipeline (processing, query)
            domain: Target domain
            metadata: Additional pipeline metadata
            
        Returns:
            Dict with registration confirmation
        """
        pipeline_info = {
            "pipeline_id": pipeline_id,
            "pipeline_type": pipeline_type,
            "domain": domain,
            "start_time": datetime.now().isoformat(),
            "status": "registered",
            "current_stage": None,
            "stages_completed": [],
            "total_stages": 5 if pipeline_type == "processing" else 4,
            "progress_percentage": 0,
            "metadata": metadata or {},
            "performance_metrics": {
                "total_duration": 0,
                "stage_durations": {},
                "bottlenecks": [],
                "throughput_metrics": {}
            }
        }
        
        self.active_pipelines[pipeline_id] = pipeline_info
        
        # Send registration event
        await self._broadcast_event(StreamingEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            event_type="pipeline_registered",
            stage="initialization",
            pipeline_id=pipeline_id,
            data={
                "pipeline_type": pipeline_type,
                "domain": domain,
                "total_stages": pipeline_info["total_stages"]
            },
            metadata=metadata or {}
        ))
        
        print(f"📡 Pipeline registered: {pipeline_id} ({pipeline_type})")
        return {"success": True, "pipeline_id": pipeline_id}

    async def start_pipeline_monitoring(
        self,
        pipeline_id: str,
        initial_data: Dict[str, Any] = None
    ) -> None:
        """Start monitoring a registered pipeline"""
        if pipeline_id not in self.active_pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not registered")
        
        pipeline_info = self.active_pipelines[pipeline_id]
        pipeline_info["status"] = "running"
        pipeline_info["actual_start_time"] = datetime.now().isoformat()
        
        await self._broadcast_event(StreamingEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            event_type="pipeline_start",
            stage="initialization",
            pipeline_id=pipeline_id,
            data=initial_data or {},
            metadata={"total_stages": pipeline_info["total_stages"]}
        ))
        
        print(f"🚀 Pipeline monitoring started: {pipeline_id}")

    async def report_stage_start(
        self,
        pipeline_id: str,
        stage: str,
        stage_data: Dict[str, Any] = None
    ) -> None:
        """Report that a stage has started"""
        if pipeline_id not in self.active_pipelines:
            print(f"⚠️  Pipeline {pipeline_id} not registered for stage start report")
            return
        
        pipeline_info = self.active_pipelines[pipeline_id]
        pipeline_info["current_stage"] = stage
        pipeline_info["performance_metrics"]["stage_durations"][stage] = {
            "start_time": datetime.now().isoformat(),
            "duration": None
        }
        
        # Calculate progress
        stage_numbers = {"01": 1, "02": 2, "03": 3, "04": 4, "05": 5, 
                        "06": 1, "07": 2, "08": 3, "09": 4}
        current_stage_num = stage_numbers.get(stage, 0)
        pipeline_info["progress_percentage"] = int((current_stage_num / pipeline_info["total_stages"]) * 100)
        
        await self._broadcast_event(StreamingEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            event_type="stage_start",
            stage=stage,
            pipeline_id=pipeline_id,
            data=stage_data or {},
            metadata={
                "progress_percentage": pipeline_info["progress_percentage"],
                "stages_completed": len(pipeline_info["stages_completed"])
            }
        ))
        
        print(f"🔄 Stage started: {pipeline_id} - {stage}")

    async def report_stage_progress(
        self,
        pipeline_id: str,
        stage: str,
        progress_data: Dict[str, Any]
    ) -> None:
        """Report progress within a stage"""
        await self._broadcast_event(StreamingEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            event_type="stage_progress",
            stage=stage,
            pipeline_id=pipeline_id,
            data=progress_data,
            metadata={}
        ))
        
        # Optional: Print significant progress updates
        if progress_data.get("percentage", 0) % 25 == 0:  # Every 25%
            print(f"📊 Stage progress: {pipeline_id} - {stage} ({progress_data.get('percentage', 0)}%)")

    async def report_stage_complete(
        self,
        pipeline_id: str,
        stage: str,
        stage_results: Dict[str, Any]
    ) -> None:
        """Report that a stage has completed"""
        if pipeline_id not in self.active_pipelines:
            print(f"⚠️  Pipeline {pipeline_id} not registered for stage completion report")
            return
        
        pipeline_info = self.active_pipelines[pipeline_id]
        pipeline_info["stages_completed"].append(stage)
        
        # Calculate stage duration
        if stage in pipeline_info["performance_metrics"]["stage_durations"]:
            start_time_str = pipeline_info["performance_metrics"]["stage_durations"][stage]["start_time"]
            start_time = datetime.fromisoformat(start_time_str)
            duration = (datetime.now() - start_time).total_seconds()
            pipeline_info["performance_metrics"]["stage_durations"][stage]["duration"] = round(duration, 2)
        
        # Update progress
        progress = int((len(pipeline_info["stages_completed"]) / pipeline_info["total_stages"]) * 100)
        pipeline_info["progress_percentage"] = progress
        
        await self._broadcast_event(StreamingEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            event_type="stage_complete",
            stage=stage,
            pipeline_id=pipeline_id,
            data=stage_results,
            metadata={
                "progress_percentage": progress,
                "stage_duration": pipeline_info["performance_metrics"]["stage_durations"].get(stage, {}).get("duration"),
                "stages_completed": len(pipeline_info["stages_completed"])
            }
        ))
        
        print(f"✅ Stage completed: {pipeline_id} - {stage} (Progress: {progress}%)")

    async def report_pipeline_complete(
        self,
        pipeline_id: str,
        final_results: Dict[str, Any]
    ) -> None:
        """Report that the entire pipeline has completed"""
        if pipeline_id not in self.active_pipelines:
            print(f"⚠️  Pipeline {pipeline_id} not registered for completion report")
            return
        
        pipeline_info = self.active_pipelines[pipeline_id]
        pipeline_info["status"] = "completed"
        pipeline_info["end_time"] = datetime.now().isoformat()
        pipeline_info["progress_percentage"] = 100
        
        # Calculate total duration
        if "actual_start_time" in pipeline_info:
            start_time = datetime.fromisoformat(pipeline_info["actual_start_time"])
            total_duration = (datetime.now() - start_time).total_seconds()
            pipeline_info["performance_metrics"]["total_duration"] = round(total_duration, 2)
        
        # Identify bottlenecks
        stage_durations = pipeline_info["performance_metrics"]["stage_durations"]
        if stage_durations:
            avg_duration = sum(s.get("duration", 0) for s in stage_durations.values()) / len(stage_durations)
            bottlenecks = [
                stage for stage, info in stage_durations.items() 
                if info.get("duration", 0) > avg_duration * 1.5
            ]
            pipeline_info["performance_metrics"]["bottlenecks"] = bottlenecks
        
        await self._broadcast_event(StreamingEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            event_type="pipeline_complete",
            stage="completion",
            pipeline_id=pipeline_id,
            data=final_results,
            metadata={
                "total_duration": pipeline_info["performance_metrics"]["total_duration"],
                "bottlenecks": pipeline_info["performance_metrics"]["bottlenecks"]
            }
        ))
        
        print(f"🎉 Pipeline completed: {pipeline_id} (Duration: {pipeline_info['performance_metrics']['total_duration']}s)")

    async def report_error(
        self,
        pipeline_id: str,
        stage: str,
        error: str,
        error_details: Dict[str, Any] = None
    ) -> None:
        """Report an error during pipeline execution"""
        if pipeline_id in self.active_pipelines:
            pipeline_info = self.active_pipelines[pipeline_id]
            pipeline_info["status"] = "failed"
            pipeline_info["error"] = error
            pipeline_info["failed_stage"] = stage
        
        await self._broadcast_event(StreamingEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            event_type="error",
            stage=stage,
            pipeline_id=pipeline_id,
            data=error_details or {},
            metadata={"error": error}
        ))
        
        print(f"❌ Pipeline error: {pipeline_id} - {stage}: {error}")

    async def _broadcast_event(self, event: StreamingEvent) -> None:
        """Broadcast event to all subscribers"""
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history.pop(0)
        
        # Broadcast to WebSocket subscribers
        if self.event_subscribers:
            event_json = json.dumps(asdict(event))
            disconnected_clients = set()
            
            for websocket in self.event_subscribers:
                try:
                    await websocket.send(event_json)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(websocket)
                except Exception as e:
                    print(f"⚠️  Error sending to client: {e}")
                    disconnected_clients.add(websocket)
            
            # Remove disconnected clients
            self.event_subscribers -= disconnected_clients

    async def get_pipeline_status(self, pipeline_id: str = None) -> Dict[str, Any]:
        """Get current status of pipeline(s)"""
        if pipeline_id:
            return self.active_pipelines.get(pipeline_id, {"error": "Pipeline not found"})
        else:
            return {
                "active_pipelines": list(self.active_pipelines.keys()),
                "pipeline_count": len(self.active_pipelines),
                "pipelines": self.active_pipelines
            }

    async def get_event_history(self, pipeline_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get event history"""
        events = self.event_history
        
        if pipeline_id:
            events = [e for e in events if e.pipeline_id == pipeline_id]
        
        # Return most recent events first
        events = sorted(events, key=lambda x: x.timestamp, reverse=True)
        return [asdict(event) for event in events[:limit]]

    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections for streaming"""
        print(f"📡 New WebSocket connection: {websocket.remote_address}")
        self.event_subscribers.add(websocket)
        
        try:
            # Send current pipeline status to new client
            status = await self.get_pipeline_status()
            await websocket.send(json.dumps({
                "type": "initial_status",
                "data": status,
                "timestamp": datetime.now().isoformat()
            }))
            
            # Keep connection alive and handle messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_websocket_message(websocket, data)
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format"
                    }))
                except Exception as e:
                    await websocket.send(json.dumps({
                        "type": "error", 
                        "message": str(e)
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            print(f"📡 WebSocket connection closed: {websocket.remote_address}")
        finally:
            self.event_subscribers.discard(websocket)

    async def _handle_websocket_message(self, websocket, data: Dict[str, Any]) -> None:
        """Handle incoming WebSocket messages"""
        message_type = data.get("type")
        
        if message_type == "get_status":
            pipeline_id = data.get("pipeline_id")
            status = await self.get_pipeline_status(pipeline_id)
            await websocket.send(json.dumps({
                "type": "status_response",
                "data": status
            }))
            
        elif message_type == "get_history":
            pipeline_id = data.get("pipeline_id")
            limit = data.get("limit", 100)
            history = await self.get_event_history(pipeline_id, limit)
            await websocket.send(json.dumps({
                "type": "history_response",
                "data": history
            }))

    async def start_websocket_server(self, host: str = "localhost", port: int = 8765) -> None:
        """Start WebSocket server for streaming"""
        print(f"🚀 Starting WebSocket server on {host}:{port}")
        
        async with websockets.serve(self.websocket_handler, host, port):
            print(f"📡 WebSocket server running on ws://{host}:{port}")
            await asyncio.Future()  # Run forever


# Global monitor instance
global_monitor = StreamingMonitor()

async def get_global_monitor() -> StreamingMonitor:
    """Get the global streaming monitor instance"""
    return global_monitor


async def main():
    """Main entry point for streaming monitor"""
    parser = argparse.ArgumentParser(
        description="Streaming Monitor - Real-time Pipeline Progress Events"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="WebSocket server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="WebSocket server port"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode with simulated events"
    )
    
    args = parser.parse_args()
    
    monitor = StreamingMonitor()
    
    if args.demo:
        # Run demo mode
        print("🎭 Running in demo mode...")
        
        # Register a demo pipeline
        await monitor.register_pipeline(
            pipeline_id="demo-pipeline-001",
            pipeline_type="query",
            domain="demo",
            metadata={"demo": True}
        )
        
        # Simulate pipeline execution
        await asyncio.sleep(1)
        await monitor.start_pipeline_monitoring("demo-pipeline-001", {"query": "What is machine learning?"})
        
        stages = ["06", "07", "08", "09"]
        stage_names = ["Query Analysis", "Unified Search", "Context Retrieval", "Response Generation"]
        
        for stage, name in zip(stages, stage_names):
            await monitor.report_stage_start("demo-pipeline-001", stage, {"stage_name": name})
            await asyncio.sleep(2)  # Simulate processing time
            await monitor.report_stage_complete("demo-pipeline-001", stage, {"status": "success"})
        
        await monitor.report_pipeline_complete("demo-pipeline-001", {"final_answer": "Machine learning is..."})
        
        print("🎭 Demo completed")
    else:
        # Start WebSocket server
        await monitor.start_websocket_server(args.host, args.port)


if __name__ == "__main__":
    asyncio.run(main())
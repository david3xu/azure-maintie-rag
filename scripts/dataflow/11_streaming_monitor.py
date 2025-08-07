#!/usr/bin/env python3
"""
Simple Pipeline Monitor - CODING_STANDARDS Compliant
Clean monitoring script without over-engineering WebSocket patterns.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.core.azure_service_container import ConsolidatedAzureServices


class SimpleMonitor:
    """Simple pipeline progress monitor"""

    def __init__(self):
        self.pipelines = {}

    def start_pipeline(self, pipeline_id: str, pipeline_type: str = "processing"):
        """Start monitoring a pipeline"""
        self.pipelines[pipeline_id] = {
            "id": pipeline_id,
            "type": pipeline_type,
            "start_time": datetime.now(),
            "status": "running",
            "progress": 0.0,
        }
        print(f"📊 Started monitoring pipeline: {pipeline_id}")
        return True

    def update_progress(self, pipeline_id: str, progress: float, message: str = ""):
        """Update pipeline progress"""
        if pipeline_id in self.pipelines:
            self.pipelines[pipeline_id]["progress"] = progress
            self.pipelines[pipeline_id]["last_update"] = datetime.now()
            print(f"🔄 {pipeline_id}: {progress:.1f}% - {message}")
            return True
        return False

    def complete_pipeline(self, pipeline_id: str):
        """Mark pipeline as complete"""
        if pipeline_id in self.pipelines:
            pipeline = self.pipelines[pipeline_id]
            pipeline["status"] = "completed"
            pipeline["end_time"] = datetime.now()
            duration = (pipeline["end_time"] - pipeline["start_time"]).total_seconds()
            print(f"✅ Pipeline {pipeline_id} completed in {duration:.1f}s")
            return True
        return False

    def get_status(self) -> dict:
        """Get monitoring status"""
        active = sum(1 for p in self.pipelines.values() if p["status"] == "running")
        completed = sum(
            1 for p in self.pipelines.values() if p["status"] == "completed"
        )

        return {
            "active_pipelines": active,
            "completed_pipelines": completed,
            "total_pipelines": len(self.pipelines),
        }


async def monitor_pipeline(pipeline_id: str = "test-pipeline"):
    """Simple pipeline monitoring demo"""
    print("📊 Simple Pipeline Monitor")

    monitor = SimpleMonitor()

    # Start pipeline
    monitor.start_pipeline(pipeline_id)

    # Simulate progress updates
    for i in range(0, 101, 20):
        await asyncio.sleep(0.5)
        monitor.update_progress(pipeline_id, i, f"Processing step {i//20 + 1}")

    # Complete pipeline
    monitor.complete_pipeline(pipeline_id)

    # Show final status
    status = monitor.get_status()
    print(f"📊 Final status: {status}")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple pipeline monitor")
    parser.add_argument(
        "--pipeline-id", default="test-pipeline", help="Pipeline ID to monitor"
    )
    args = parser.parse_args()

    result = asyncio.run(monitor_pipeline(args.pipeline_id))
    sys.exit(0 if result else 1)

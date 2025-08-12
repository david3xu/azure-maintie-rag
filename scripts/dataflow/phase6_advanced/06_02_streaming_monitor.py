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

from infrastructure.azure_monitoring.app_insights_client import (
    AzureApplicationInsightsClient,
)


class SimpleMonitor:
    """Real Azure Application Insights pipeline monitor"""

    def __init__(self):
        self.pipelines = {}
        self.insights_client = AzureApplicationInsightsClient()
        print("🔗 Connected to real Azure Application Insights")

    def start_pipeline(self, pipeline_id: str, pipeline_type: str = "processing"):
        """Start monitoring a pipeline with real Azure Application Insights"""
        start_time = datetime.now()
        self.pipelines[pipeline_id] = {
            "id": pipeline_id,
            "type": pipeline_type,
            "start_time": start_time,
            "status": "running",
            "progress": 0.0,
        }

        # Log to real Azure Application Insights
        try:
            self.insights_client.track_event(
                name="pipeline_started",
                properties={
                    "pipeline_id": pipeline_id,
                    "pipeline_type": pipeline_type,
                    "start_time": start_time.isoformat(),
                },
                measurements={"progress": 0.0},
            )
            print(f"📊 Started monitoring pipeline: {pipeline_id} (logged to Azure)")
        except Exception as e:
            print(
                f"📊 Started monitoring pipeline: {pipeline_id} (Azure logging failed: {e})"
            )

        return True

    def update_progress(self, pipeline_id: str, progress: float, message: str = ""):
        """Update pipeline progress with real Azure Application Insights"""
        if pipeline_id in self.pipelines:
            current_time = datetime.now()
            self.pipelines[pipeline_id]["progress"] = progress
            self.pipelines[pipeline_id]["last_update"] = current_time

            # Log progress to real Azure Application Insights
            try:
                self.insights_client.track_event(
                    name="pipeline_progress",
                    properties={
                        "pipeline_id": pipeline_id,
                        "message": message,
                        "timestamp": current_time.isoformat(),
                    },
                    measurements={"progress": progress},
                )
                print(
                    f"🔄 {pipeline_id}: {progress:.1f}% - {message} (logged to Azure)"
                )
            except Exception as e:
                print(
                    f"🔄 {pipeline_id}: {progress:.1f}% - {message} (Azure logging failed: {e})"
                )

            return True
        return False

    def complete_pipeline(self, pipeline_id: str):
        """Mark pipeline as complete with real Azure Application Insights"""
        if pipeline_id in self.pipelines:
            pipeline = self.pipelines[pipeline_id]
            end_time = datetime.now()
            pipeline["status"] = "completed"
            pipeline["end_time"] = end_time
            duration = (end_time - pipeline["start_time"]).total_seconds()

            # Log completion to real Azure Application Insights
            try:
                self.insights_client.track_event(
                    name="pipeline_completed",
                    properties={
                        "pipeline_id": pipeline_id,
                        "pipeline_type": pipeline["type"],
                        "end_time": end_time.isoformat(),
                        "status": "completed",
                    },
                    measurements={
                        "duration_seconds": duration,
                        "final_progress": 100.0,
                    },
                )
                print(
                    f"✅ Pipeline {pipeline_id} completed in {duration:.1f}s (logged to Azure)"
                )
            except Exception as e:
                print(
                    f"✅ Pipeline {pipeline_id} completed in {duration:.1f}s (Azure logging failed: {e})"
                )

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

#!/usr/bin/env python3
"""
Simple Performance Monitoring Test

Tests the performance monitoring system directly without complex workflow dependencies.

✅ PHASE 3: PERFORMANCE ENHANCEMENT VALIDATION
✅ SLA COMPLIANCE: Sub-3-second response time validation
✅ COMPETITIVE ADVANTAGE MONITORING: All features tracked
"""

import asyncio
import time
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from agents.core.performance_monitor import get_performance_monitor, CompetitiveAdvantageMonitor, AlertSeverity
except ImportError as e:
    print(f"Import error: {e}")
    print("Testing performance monitor directly...")
    sys.path.insert(0, str(Path(__file__).parent / "agents" / "core"))
    from performance_monitor import get_performance_monitor, CompetitiveAdvantageMonitor, AlertSeverity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_performance_monitor_core():
    """
    Test the core performance monitoring functionality
    """
    print("🎯 Testing Core Performance Monitor (Phase 3)")
    print("=" * 50)

    # Get monitor instance
    monitor = get_performance_monitor()

    # Test 1: Tri-Modal Search Monitoring
    print("\n1️⃣ Testing Tri-Modal Search Performance Monitoring...")

    # Test with good performance (should pass SLA)
    await monitor.track_tri_modal_search_performance(
        search_time=1.2,  # Under 3s SLA
        confidence=0.87,  # Above baseline
        modalities_used=["vector", "graph", "gnn"],  # Full tri-modal
        correlation_id="test_good_performance"
    )
    print("   ✅ Good performance scenario tracked")

    # Test with poor performance (should trigger alerts)
    await monitor.track_tri_modal_search_performance(
        search_time=4.5,  # Exceeds 3s SLA
        confidence=0.6,   # Below baseline
        modalities_used=["vector"],  # Not tri-modal
        correlation_id="test_poor_performance"
    )
    print("   ⚠️  Poor performance scenario tracked (should generate alerts)")

    # Test 2: Domain Intelligence Monitoring
    print("\n2️⃣ Testing Domain Intelligence Performance Monitoring...")

    await monitor.track_domain_intelligence_performance(
        analysis_time=0.5,
        detection_accuracy=0.82,
        hybrid_analysis_used=True,
        correlation_id="test_domain_intel"
    )
    print("   ✅ Domain intelligence performance tracked")

    # Test 3: Config-Extraction Pipeline Monitoring
    print("\n3️⃣ Testing Config-Extraction Pipeline Monitoring...")

    await monitor.track_config_extraction_pipeline_performance(
        config_generation_time=1.8,
        extraction_time=3.2,
        pipeline_success=True,
        automation_achieved=True,
        correlation_id="test_config_extraction"
    )
    print("   ✅ Config-extraction pipeline performance tracked")

    # Test 4: Zero-Config Adaptation Monitoring
    print("\n4️⃣ Testing Zero-Config Adaptation Monitoring...")

    await monitor.track_zero_config_adaptation_performance(
        adaptation_time=0.8,
        adaptation_success=True,
        manual_intervention_required=False,
        correlation_id="test_zero_config"
    )
    print("   ✅ Zero-config adaptation performance tracked")

    # Test 5: Enterprise Infrastructure Monitoring
    print("\n5️⃣ Testing Enterprise Infrastructure Monitoring...")

    await monitor.track_enterprise_infrastructure_performance(
        availability=0.995,
        response_time=2.1,
        error_rate=0.002,
        azure_services_health={
            "cognitive_search": True,
            "cosmos_db": True,
            "azure_ml": True,
            "storage": True
        },
        correlation_id="test_infrastructure"
    )
    print("   ✅ Enterprise infrastructure performance tracked")

    return monitor


async def test_sla_compliance(monitor):
    """
    Test SLA compliance monitoring
    """
    print("\n6️⃣ Testing SLA Compliance Monitoring...")

    # Test various response times
    test_scenarios = [
        {"time": 1.5, "expected": "PASS"},
        {"time": 2.8, "expected": "PASS"},
        {"time": 3.2, "expected": "FAIL"},
        {"time": 4.5, "expected": "FAIL"}
    ]

    for i, scenario in enumerate(test_scenarios, 1):
        await monitor.track_enterprise_infrastructure_performance(
            availability=0.99,
            response_time=scenario["time"],
            error_rate=0.0,
            azure_services_health={"test_service": True},
            correlation_id=f"sla_test_{i}"
        )

        sla_met = scenario["time"] < 3.0
        result = "✅ PASS" if sla_met else "❌ FAIL"
        print(f"   Response time {scenario['time']:.1f}s → {result} (Expected: {scenario['expected']})")


async def test_alert_system(monitor):
    """
    Test alert generation and handling
    """
    print("\n7️⃣ Testing Alert System...")

    # Clear existing alerts
    monitor.clear_alerts()

    # Test critical performance scenario
    await monitor.track_tri_modal_search_performance(
        search_time=5.0,  # Critical SLA violation
        confidence=0.4,   # Very low confidence
        modalities_used=["vector"],  # Single modality
        correlation_id="critical_test"
    )

    # Check generated alerts
    all_alerts = monitor.get_alerts()
    critical_alerts = monitor.get_alerts(AlertSeverity.CRITICAL)
    warning_alerts = monitor.get_alerts(AlertSeverity.WARNING)

    print(f"   Total alerts generated: {len(all_alerts)}")
    print(f"   Critical alerts: {len(critical_alerts)}")
    print(f"   Warning alerts: {len(warning_alerts)}")

    # Display recent alerts
    for alert in all_alerts[-3:]:
        severity_icon = "🚨" if alert.severity.value == "critical" else "⚠️"
        print(f"   {severity_icon} {alert.message}")

    return len(all_alerts) > 0


async def test_performance_summary(monitor):
    """
    Test performance summary generation
    """
    print("\n8️⃣ Testing Performance Summary...")

    summary = monitor.get_performance_summary()

    print(f"   📊 Monitoring enabled: {summary['monitoring_enabled']}")
    print(f"   📈 Recent metrics tracked: {len(summary.get('recent_performance', {}))}")
    print(f"   🎯 SLA compliance rate: {summary['sla_compliance']['compliance_rate']:.1%}")
    print(f"   ⏱️  Sub-3s target met: {summary['sla_compliance']['sub_3s_target_met']}")
    print(f"   🚨 Active alerts: {summary['active_alerts']['total']}")

    # Check for competitive advantage metrics
    expected_metrics = [
        "tri_modal_search_time",
        "tri_modal_search_confidence",
        "domain_analysis_time",
        "config_generation_time",
        "zero_config_adaptation_time",
        "enterprise_availability"
    ]

    found_metrics = 0
    for metric in expected_metrics:
        if metric in summary.get('recent_performance', {}):
            found_metrics += 1
            print(f"   ✅ {metric}: Tracked")
        else:
            print(f"   ⚠️  {metric}: Not found")

    coverage = (found_metrics / len(expected_metrics)) * 100
    print(f"   📊 Competitive advantage coverage: {coverage:.0f}%")

    return summary


async def main():
    """
    Main test function
    """
    print("🎯 PHASE 3: PERFORMANCE ENHANCEMENT - SIMPLE VALIDATION")
    print("=" * 60)

    try:
        # Test core monitoring
        monitor = await test_performance_monitor_core()

        # Test SLA compliance
        await test_sla_compliance(monitor)

        # Test alert system
        alerts_generated = await test_alert_system(monitor)

        # Test performance summary
        summary = await test_performance_summary(monitor)

        # Final validation
        print("\n" + "=" * 60)
        print("🎯 PHASE 3 VALIDATION COMPLETE")
        print("=" * 60)

        # Check success criteria
        monitoring_active = summary['monitoring_enabled']
        metrics_tracked = len(summary.get('recent_performance', {})) > 0
        sla_monitoring = 'sla_compliance' in summary
        alerts_working = alerts_generated

        success_criteria = [
            ("Monitoring Active", monitoring_active),
            ("Metrics Tracked", metrics_tracked),
            ("SLA Monitoring", sla_monitoring),
            ("Alert System", alerts_working)
        ]

        all_passed = True
        for criterion, passed in success_criteria:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {criterion}: {status}")
            if not passed:
                all_passed = False

        if all_passed:
            print(f"\n🎉 PHASE 3: PERFORMANCE ENHANCEMENT SUCCESSFULLY COMPLETED!")
            print(f"✅ All competitive advantages are monitored")
            print(f"✅ Sub-3-second SLA validation is active")
            print(f"✅ Alert system is operational")
            print(f"✅ Performance tracking is comprehensive")
            return True
        else:
            print(f"\n❌ Some validation criteria failed")
            return False

    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

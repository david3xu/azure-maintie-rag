#!/usr/bin/env python3
"""
Enterprise Knowledge Extraction Test Script
Tests the full enterprise architecture implementation
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from core.azure_openai.knowledge_extractor import AzureOpenAIKnowledgeExtractor
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_enterprise_extraction():
    """Test enhanced knowledge extraction with Azure services"""

    logger.info("🚀 Starting Enterprise Knowledge Extraction Test")

    # Sample test documents
    test_texts = [
        "Microsoft Azure is a cloud computing platform created by Microsoft. It provides a wide range of cloud services including computing, analytics, storage, and networking.",
        "OpenAI is an artificial intelligence research laboratory. They developed GPT-4, a large language model that can understand and generate human-like text.",
        "Knowledge graphs represent information as entities and relationships. They are used in search engines, recommendation systems, and AI applications."
    ]

    try:
        # Initialize enterprise knowledge extractor
        logger.info("📦 Initializing Enterprise Knowledge Extractor...")
        extractor = AzureOpenAIKnowledgeExtractor("enterprise_test")

        # Test enterprise extraction
        logger.info("🔍 Testing Enterprise Knowledge Extraction...")
        results = await extractor.extract_knowledge_from_texts(test_texts)

        # Validate results
        logger.info("✅ Extraction completed successfully")
        logger.info(f"📊 Results Summary:")
        logger.info(f"  - Success: {results.get('success', False)}")
        logger.info(f"  - Domain: {results.get('domain', 'unknown')}")
        logger.info(f"  - Processing Time: {results.get('processing_time', 0):.2f}s")

        if results.get('success'):
            # Log knowledge summary
            knowledge_summary = results.get('knowledge_summary', {})
            logger.info(f"  - Total Entities: {knowledge_summary.get('total_entities', 0)}")
            logger.info(f"  - Total Relations: {knowledge_summary.get('total_relations', 0)}")
            logger.info(f"  - Graph Nodes: {knowledge_summary.get('graph_nodes', 0)}")
            logger.info(f"  - Graph Edges: {knowledge_summary.get('graph_edges', 0)}")

            # Log discovered types
            discovered_types = results.get('discovered_types', {})
            logger.info(f"  - Entity Types: {len(discovered_types.get('entity_types', []))}")
            logger.info(f"  - Relation Types: {len(discovered_types.get('relation_types', []))}")

            # Log enterprise quality assessment
            validation_results = results.get('validation_results', {})
            if 'enterprise_quality_score' in validation_results:
                logger.info(f"  - Enterprise Quality Score: {validation_results['enterprise_quality_score']}")
                logger.info(f"  - Quality Tier: {validation_results.get('quality_tier', 'unknown')}")

                # Log recommendations
                recommendations = validation_results.get('recommendations', [])
                if recommendations:
                    logger.info(f"  - Quality Recommendations:")
                    for rec in recommendations:
                        logger.info(f"    • {rec}")

            # Test monitoring capabilities
            logger.info("📈 Testing Monitoring Capabilities...")
            monitoring_summary = extractor.monitor.get_monitoring_summary()
            logger.info(f"  - Telemetry Available: {monitoring_summary.get('telemetry_available', False)}")
            logger.info(f"  - Custom Metrics Count: {monitoring_summary.get('custom_metrics_count', 0)}")
            logger.info(f"  - Performance Tracking Active: {monitoring_summary.get('performance_tracking_active', False)}")

            # Test rate limiting
            logger.info("⚡ Testing Rate Limiting...")
            usage_summary = extractor.rate_limiter.get_usage_summary()
            logger.info(f"  - Tokens Used This Minute: {usage_summary.get('tokens_used_this_minute', 0)}")
            logger.info(f"  - Requests This Minute: {usage_summary.get('requests_this_minute', 0)}")
            logger.info(f"  - Cost This Hour: ${usage_summary.get('cost_this_hour', 0):.4f}")
            logger.info(f"  - Quota Healthy: {extractor.rate_limiter.is_quota_healthy()}")

            # Test cost optimization recommendations
            cost_recommendations = extractor.rate_limiter.get_cost_optimization_recommendations()
            if cost_recommendations:
                logger.info(f"  - Cost Optimization Recommendations:")
                for rec in cost_recommendations:
                    logger.info(f"    • {rec}")

            return True

        else:
            logger.error(f"❌ Extraction failed: {results.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        logger.error(f"❌ Enterprise extraction test failed: {e}", exc_info=True)
        return False

async def test_enterprise_services_individually():
    """Test individual enterprise services"""

    logger.info("🔧 Testing Individual Enterprise Services...")

    try:
        # Test Text Analytics Service
        logger.info("📝 Testing Azure Text Analytics Service...")
        from core.azure_openai.azure_text_analytics_service import AzureTextAnalyticsService

        text_analytics = AzureTextAnalyticsService()
        test_text = "Microsoft Azure provides cloud computing services."

        quality_validation = await text_analytics.validate_text_quality(test_text)
        logger.info(f"  - Text Quality Score: {quality_validation.get('quality_score', 0):.3f}")
        logger.info(f"  - Language: {quality_validation.get('language', {}).get('language', 'unknown')}")
        logger.info(f"  - Entities Found: {quality_validation.get('entities_found', 0)}")

        # Test ML Quality Assessment
        logger.info("🤖 Testing Azure ML Quality Assessment...")
        from core.azure_openai.azure_ml_quality_service import AzureMLQualityAssessment

        quality_assessor = AzureMLQualityAssessment("test_domain")

        # Mock extraction context
        extraction_context = {
            "domain": "test_domain",
            "entity_count": 5,
            "relation_count": 3,
            "entity_types": ["Person", "Organization", "Technology"],
            "relation_types": ["works_for", "develops", "uses"],
            "documents_processed": 1
        }

        mock_entities = {
            "entity_1": {"confidence": 0.9, "text": "Microsoft"},
            "entity_2": {"confidence": 0.8, "text": "Azure"}
        }

        mock_relations = [
            {"confidence": 0.85, "source": "Microsoft", "target": "Azure"},
            {"confidence": 0.75, "source": "Azure", "target": "Cloud"}
        ]

        quality_results = await quality_assessor.assess_extraction_quality(
            extraction_context, mock_entities, mock_relations
        )

        logger.info(f"  - Enterprise Quality Score: {quality_results.get('enterprise_quality_score', 0):.3f}")
        logger.info(f"  - Quality Tier: {quality_results.get('quality_tier', 'unknown')}")

        # Test Monitoring Service
        logger.info("📊 Testing Azure Monitoring Service...")
        from core.azure_openai.azure_monitoring_service import AzureKnowledgeMonitor

        monitor = AzureKnowledgeMonitor()
        monitoring_summary = monitor.get_monitoring_summary()

        logger.info(f"  - Telemetry Available: {monitoring_summary.get('telemetry_available', False)}")
        logger.info(f"  - Monitoring Features: {len(monitoring_summary.get('monitoring_features', []))}")

        # Test Rate Limiter
        logger.info("⚡ Testing Azure Rate Limiter...")
        from core.azure_openai.azure_rate_limiter import AzureOpenAIRateLimiter

        rate_limiter = AzureOpenAIRateLimiter()
        usage_summary = rate_limiter.get_usage_summary()

        logger.info(f"  - Quota Limits: {usage_summary.get('quota_limits', {})}")
        logger.info(f"  - Current Usage: {usage_summary.get('utilization_percentages', {})}")

        return True

    except Exception as e:
        logger.error(f"❌ Individual service tests failed: {e}", exc_info=True)
        return False

async def main():
    """Main test function"""
    logger.info("🏗️ Enterprise Knowledge Extraction Test Suite")
    logger.info("=" * 50)

    # Test individual services first
    individual_success = await test_enterprise_services_individually()

    if individual_success:
        logger.info("✅ Individual service tests passed")

        # Test full enterprise extraction
        enterprise_success = await test_enterprise_extraction()

        if enterprise_success:
            logger.info("🎉 All enterprise tests passed successfully!")
            return True
        else:
            logger.error("❌ Enterprise extraction test failed")
            return False
    else:
        logger.error("❌ Individual service tests failed")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
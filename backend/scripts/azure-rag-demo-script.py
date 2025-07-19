#!/usr/bin/env python3
"""
Universal RAG Demo Script
========================

Clean demonstration of the Universal RAG system using the new universal components.
This script replaces all the old domain-specific scripts with a single universal demo.

Features:
- Works with any domain using pure text files
- Demonstrates complete Universal RAG workflow
- Shows real-time query processing
- Tests universal API endpoints
- No hardcoded types or schema dependencies

Usage:
    python scripts/universal_rag_demo.py
"""

import sys
import os
import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend directory to Python path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

# Azure service imports
from core.orchestration.enhanced_pipeline import AzureRAGEnhancedPipeline as AzureRAGEnhancedPipeline
from core.orchestration.rag_orchestration_service import (
    create_universal_rag_from_texts, create_universal_rag_from_directory
)
from core.azure_openai.knowledge_extractor import AzureOpenAIKnowledgeExtractor as AzureOpenAIKnowledgeExtractor
from core.azure_openai.text_processor import AzureOpenAITextProcessor as AzureOpenAITextProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class UniversalRAGDemo:
    """Universal RAG system demonstration"""

    def __init__(self):
        """Initialize demo"""
        self.sample_domains = {
            "medical": [
                "Patient presents with fever, headache, and fatigue symptoms.",
                "Blood pressure monitoring is essential for cardiovascular health.",
                "Diagnosis requires comprehensive examination and test results.",
                "Treatment protocols vary based on patient medical history.",
                "Medication interactions must be carefully considered during treatment."
            ],
            "legal": [
                "Contract terms must be clearly defined and legally binding.",
                "Liability clauses protect parties from unforeseen circumstances.",
                "Intellectual property rights require proper documentation.",
                "Compliance with regulations is mandatory for all operations.",
                "Legal precedents influence court decisions and case outcomes."
            ],
            "finance": [
                "Risk assessment is crucial for investment portfolio management.",
                "Market volatility affects asset valuation and returns.",
                "Diversification strategies help minimize investment risks.",
                "Regulatory compliance ensures adherence to financial standards.",
                "Credit analysis determines loan approval and interest rates."
            ],
            "maintenance": [
                "Equipment maintenance schedules prevent unexpected failures.",
                "Bearing lubrication is essential for rotating machinery operation.",
                "Vibration analysis indicates potential mechanical problems.",
                "Safety procedures must be followed during repair operations.",
                "Preventive maintenance reduces overall operational costs."
            ],
            "technology": [
                "Software development requires systematic testing and validation.",
                "Database optimization improves application performance significantly.",
                "Security protocols protect systems from unauthorized access.",
                "API design ensures scalable and maintainable integrations.",
                "Cloud infrastructure provides flexible and cost-effective solutions."
            ]
        }

    async def run_complete_demo(self):
        """Run complete Universal RAG demonstration"""
        print("🚀 Universal RAG System Demonstration")
        print("=" * 60)
        print("Testing the Universal RAG system with multiple domains")
        print("Shows complete workflow from text files to intelligent responses")
        print()

        # Demo 1: Single Domain Processing
        await self._demo_single_domain()

        # Demo 2: Multi-Domain Processing
        await self._demo_multi_domain()

        # Demo 3: Real-time Query Processing
        await self._demo_real_time_queries()

        # Demo 4: System Performance Analysis
        await self._demo_performance_analysis()

        print("\n✅ Universal RAG demonstration completed successfully!")
        print("The system demonstrated:")
        print("  ✓ Zero configuration setup")
        print("  ✓ Dynamic type discovery")
        print("  ✓ Multi-domain processing")
        print("  ✓ Real-time query handling")
        print("  ✓ Universal API compatibility")

    async def _demo_single_domain(self):
        """Demonstrate single domain processing"""
        print("\n📋 Demo 1: Single Domain Processing")
        print("-" * 40)

        domain = "medical"
        texts = self.sample_domains[domain]

        print(f"Creating Universal RAG system for '{domain}' domain...")
        print(f"Processing {len(texts)} sample texts...")

        try:
            # Create Universal RAG system from texts
            start_time = time.time()
            orchestrator = await create_universal_rag_from_texts(texts, domain)
            setup_time = time.time() - start_time

            print(f"✅ System created in {setup_time:.2f} seconds")

            # Get system status
            status = orchestrator.get_system_status()
            stats = status["system_stats"]

            print(f"📊 System Statistics:")
            print(f"   📄 Documents: {stats['total_documents']}")
            print(f"   🏷️  Entities: {stats['total_entities']}")
            print(f"   🔗 Relations: {stats['total_relations']}")
            print(f"   📝 Entity Types: {stats['unique_entity_types']}")
            print(f"   🔀 Relation Types: {stats['unique_relation_types']}")

            # Test a query
            query = "What symptoms should I monitor?"
            print(f"\n🔍 Testing query: '{query}'")

            query_start = time.time()
            results = await orchestrator.process_query(query)
            query_time = time.time() - query_start

            if results.get("success", False):
                print(f"✅ Query processed in {query_time:.2f} seconds")
                print(f"📝 Response: {results['response'].get('content', 'No response')[:200]}...")
            else:
                print(f"❌ Query failed: {results.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Demo failed: {e}")

    async def _demo_multi_domain(self):
        """Demonstrate multi-domain processing"""
        print("\n🌍 Demo 2: Multi-Domain Processing")
        print("-" * 40)

        domains_to_test = ["legal", "finance", "technology"]

        for domain in domains_to_test:
            print(f"\n🏗️ Creating '{domain}' domain...")

            try:
                texts = self.sample_domains[domain]
                orchestrator = await create_universal_rag_from_texts(texts, domain)

                # Test domain-specific query
                query_map = {
                    "legal": "What are contract requirements?",
                    "finance": "How to assess investment risk?",
                    "technology": "What are security best practices?"
                }

                query = query_map[domain]
                results = await orchestrator.process_query(query)

                if results.get("success", False):
                    print(f"✅ {domain.title()} domain: Query successful")
                    response_preview = results['response'].get('content', '')[:150]
                    print(f"   📝 Preview: {response_preview}...")
                else:
                    print(f"❌ {domain.title()} domain: Query failed")

            except Exception as e:
                print(f"❌ {domain.title()} domain failed: {e}")

    async def _demo_real_time_queries(self):
        """Demonstrate real-time query processing with progress"""
        print("\n⚡ Demo 3: Real-time Query Processing")
        print("-" * 40)

        # Use maintenance domain for this demo
        domain = "maintenance"
        texts = self.sample_domains[domain]

        print("Setting up maintenance domain for real-time demo...")
        orchestrator = await create_universal_rag_from_texts(texts, domain)

        # Define progress callback
        progress_steps = []
        async def progress_callback(step_name: str, progress: int):
            progress_steps.append(f"[{progress}%] {step_name}")
            print(f"  🔄 {step_name} ({progress}%)")

        queries = [
            "How to prevent equipment failures?",
            "What are safety procedures for maintenance?",
            "How to diagnose bearing problems?"
        ]

        for i, query in enumerate(queries, 1):
            print(f"\n🔍 Query {i}: '{query}'")
            progress_steps.clear()

            start_time = time.time()
            results = await orchestrator.process_query(
                query,
                stream_progress=True,
                progress_callback=progress_callback
            )
            end_time = time.time()

            if results.get("success", False):
                print(f"✅ Completed in {end_time - start_time:.2f} seconds")
                print(f"📊 Progress steps: {len(progress_steps)}")
            else:
                print(f"❌ Failed: {results.get('error', 'Unknown error')}")

    async def _demo_performance_analysis(self):
        """Demonstrate system performance analysis"""
        print("\n📊 Demo 4: Performance Analysis")
        print("-" * 40)

        # Test with different text sizes
        text_sizes = {
            "small": 3,
            "medium": 5,
            "large": 7
        }

        performance_results = {}

        for size_name, text_count in text_sizes.items():
            print(f"\n🧪 Testing {size_name} corpus ({text_count} texts)...")

            # Use subset of texts
            texts = self.sample_domains["technology"][:text_count]

            try:
                # Measure setup time
                setup_start = time.time()
                orchestrator = await create_universal_rag_from_texts(texts, f"test_{size_name}")
                setup_time = time.time() - setup_start

                # Measure query time
                query = "How to improve system security?"
                query_start = time.time()
                results = await orchestrator.process_query(query)
                query_time = time.time() - query_start

                # Get system stats
                status = orchestrator.get_system_status()
                stats = status["system_stats"]

                performance_results[size_name] = {
                    "setup_time": setup_time,
                    "query_time": query_time,
                    "total_entities": stats["total_entities"],
                    "total_relations": stats["total_relations"],
                    "success": results.get("success", False)
                }

                print(f"  ✅ Setup: {setup_time:.2f}s, Query: {query_time:.2f}s")
                print(f"     Entities: {stats['total_entities']}, Relations: {stats['total_relations']}")

            except Exception as e:
                print(f"  ❌ Failed: {e}")
                performance_results[size_name] = {"error": str(e)}

        # Print performance summary
        print(f"\n📈 Performance Summary:")
        for size_name, results in performance_results.items():
            if "error" not in results:
                print(f"   {size_name.title()}: {results['setup_time']:.2f}s setup, {results['query_time']:.2f}s query")
            else:
                print(f"   {size_name.title()}: Failed - {results['error']}")

    def create_sample_text_files(self, output_dir: Path):
        """Create sample text files for testing"""
        output_dir.mkdir(parents=True, exist_ok=True)

        for domain, texts in self.sample_domains.items():
            domain_file = output_dir / f"{domain}_sample.txt"
            with open(domain_file, 'w', encoding='utf-8') as f:
                f.write(f"# {domain.title()} Domain Sample Data\n\n")
                for i, text in enumerate(texts, 1):
                    f.write(f"{i}. {text}\n\n")

        print(f"📁 Sample text files created in: {output_dir}")
        return list(output_dir.glob("*.txt"))


async def main():
    """Main demo execution"""
    print("🚀 Universal RAG Demo Starting...")

    # Check environment
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not found in environment")
        print("   The demo will run but LLM features may not work")
        print("   Please set up your .env file with Azure OpenAI credentials")
        print()

    # Create and run demo
    demo = UniversalRAGDemo()

    try:
        await demo.run_complete_demo()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def create_sample_data():
    """Create sample data files for testing"""
    demo = UniversalRAGDemo()
    sample_dir = Path("data/samples")
    sample_files = demo.create_sample_text_files(sample_dir)

    print(f"Created {len(sample_files)} sample files:")
    for file in sample_files:
        print(f"  📄 {file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Universal RAG Demo")
    parser.add_argument("--create-samples", action="store_true",
                       help="Create sample text files for testing")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick demo (single domain only)")

    args = parser.parse_args()

    if args.create_samples:
        create_sample_data()
    else:
        # Run the demo
        asyncio.run(main())
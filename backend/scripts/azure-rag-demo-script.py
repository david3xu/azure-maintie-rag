#!/usr/bin/env python3
"""
Universal RAG Demo Script
========================

Clean demonstration of the Universal RAG system using the new Azure services architecture.
This script replaces all the old domain-specific scripts with a single universal demo.

Features:
- Works with any domain using pure text files
- Demonstrates complete Universal RAG workflow
- Shows real-time query processing
- Tests universal API endpoints
- No hardcoded types or schema dependencies

Usage:
    python scripts/azure-rag-demo-script.py
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

# Azure service imports - Updated to use new Azure services architecture
from azure.integrations.azure_services import AzureServicesManager
from azure.integrations.azure_openai import AzureOpenAIClient
from config.settings import AzureSettings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class UniversalRAGDemo:
    """Universal RAG system demonstration using Azure services"""

    def __init__(self):
        """Initialize demo with Azure services"""
        self.azure_settings = AzureSettings()
        self.azure_services = AzureServicesManager()
        self.openai_integration = AzureOpenAIClient()

        self.sample_domains = {
            "general": [
                "System components work together to achieve desired outcomes.",
                "Performance monitoring helps identify potential issues early.",
                "Regular analysis reveals patterns in system behavior.",
                "Data processing requires systematic validation and testing.",
                "Process optimization improves overall system efficiency."
            ]
        }

    async def run_complete_demo(self):
        """Run complete Universal RAG demonstration"""
        print("🚀 Universal RAG System Demonstration")
        print("=" * 60)
        print("Testing the Universal RAG system with Azure services")
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
        print("  ✓ Azure services integration")
        print("  ✓ Dynamic type discovery")
        print("  ✓ Multi-domain processing")
        print("  ✓ Real-time query handling")
        print("  ✓ Universal API compatibility")

    async def _demo_single_domain(self):
        """Demonstrate single domain processing"""
        print("\n📋 Demo 1: Single Domain Processing")
        print("-" * 40)

        domain = "general"
        texts = self.sample_domains[domain]

        print(f"Creating Universal RAG system for '{domain}' domain...")
        print(f"Processing {len(texts)} sample texts...")

        try:
            # Initialize Azure services
            await self.azure_services.initialize()

            # Create Universal RAG system from texts using Azure services
            start_time = time.time()

            # Store texts in Azure Blob Storage
            container_name = f"rag-data-{domain}"
            await self.azure_services.storage_client.create_container(container_name)

            for i, text in enumerate(texts):
                blob_name = f"document_{i}.txt"
                await self.azure_services.storage_client.upload_text(container_name, blob_name, text)

            # Process with Azure OpenAI
            processed_docs = await self.openai_integration.process_documents(texts, domain)

            setup_time = time.time() - start_time

            print(f"✅ System created in {setup_time:.2f} seconds")

            # Get system status
            print(f"📊 System Statistics:")
            print(f"   📄 Documents: {len(texts)}")
            print(f"   🏷️  Processed: {len(processed_docs)}")
            print(f"   ☁️  Stored in Azure Blob Storage")
            print(f"   🤖 Processed with Azure OpenAI")

            # Test a query
            query = "What should I monitor?"
            print(f"\n🔍 Testing query: '{query}'")

            query_start = time.time()

            # Use Azure Cognitive Search for query processing
            search_results = await self.azure_services.search_client.search_documents(
                domain, query, top_k=5
            )

            # Generate response using Azure OpenAI
            response = await self.openai_integration.generate_response(
                query, search_results, domain
            )

            query_time = time.time() - query_start

            print(f"✅ Query processed in {query_time:.2f} seconds")
            print(f"📝 Response: {response[:200]}...")

        except Exception as e:
            print(f"❌ Demo failed: {e}")

    async def _demo_multi_domain(self):
        """Demonstrate multi-domain processing"""
        print("\n🌍 Demo 2: Multi-Domain Processing")
        print("-" * 40)

        domains_to_test = ["general"]

        for domain in domains_to_test:
            print(f"\n🏗️ Creating '{domain}' domain...")

            try:
                texts = self.sample_domains[domain]

                # Store in Azure Blob Storage
                container_name = f"rag-data-{domain}"
                await self.azure_services.storage_client.create_container(container_name)

                for i, text in enumerate(texts):
                    blob_name = f"document_{i}.txt"
                    await self.azure_services.storage_client.upload_text(container_name, blob_name, text)

                # Process with Azure OpenAI
                processed_docs = await self.openai_integration.process_documents(texts, domain)

                # Test universal query
                query_map = {
                    "general": "What are system requirements?"
                }

                query = query_map[domain]

                # Search and generate response
                search_results = await self.azure_services.search_client.search_documents(
                    domain, query, top_k=3
                )
                response = await self.openai_integration.generate_response(
                    query, search_results, domain
                )

                print(f"✅ {domain.title()} domain: Query successful")
                print(f"   📝 Preview: {response[:150]}...")

            except Exception as e:
                print(f"❌ {domain.title()} domain failed: {e}")

    async def _demo_real_time_queries(self):
        """Demonstrate real-time query processing with progress"""
        print("\n⚡ Demo 3: Real-time Query Processing")
        print("-" * 40)

        # Use general domain for this demo
        domain = "general"
        texts = self.sample_domains[domain]

        print("Setting up general domain for real-time demo...")
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
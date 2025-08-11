#!/usr/bin/env python3
"""
Agent 1 Real Output Analysis - Test with Real Azure Services and Data
=====================================================================

This script tests Agent 1 (Domain Intelligence) with real Azure services and real data,
then analyzes if the actual output matches our UniversalDomainAnalysis design.

Tests:
1. Load 5 real files from data/raw/azure-ai-services-language-service_output/
2. Run Agent 1 with real Azure OpenAI service
3. Capture full UniversalDomainAnalysis output
4. Analyze field completeness and validate against centralized schema
5. Save results locally for inspection
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, '/workspace/azure-maintie-rag')

async def test_agent1_real_output():
    """Test Agent 1 with real Azure services and real data files"""
    
    print("🧪 AGENT 1 REAL OUTPUT ANALYSIS")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # 1. Load real data files
    print("📄 Step 1: Loading real data files...")
    data_dir = Path('/workspace/azure-maintie-rag/data/raw/azure-ai-services-language-service_output')
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    # Get first 5 real files
    real_files = list(data_dir.glob('*.md'))[:5]
    if len(real_files) < 5:
        print(f"⚠️  Only found {len(real_files)} files, using all available")
    
    print(f"✅ Found {len(real_files)} real Azure AI documentation files:")
    for i, file_path in enumerate(real_files, 1):
        file_size = file_path.stat().st_size
        print(f"   {i}. {file_path.name} ({file_size:,} bytes)")
    
    # 2. Initialize Agent 1 with real Azure services
    print(f"\n🔌 Step 2: Initializing Agent 1 with real Azure OpenAI...")
    try:
        from agents.domain_intelligence.agent import run_domain_analysis
        print("✅ Agent 1 imported successfully")
    except Exception as e:
        print(f"❌ Agent 1 import failed: {e}")
        return False
    
    # 3. Test Agent 1 with each real file
    print(f"\n🧠 Step 3: Testing Agent 1 with real data and Azure services...")
    
    results = {}
    total_start = time.time()
    
    for i, file_path in enumerate(real_files, 1):
        print(f"\n--- Testing File {i}: {file_path.name} ---")
        
        # Load content
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            content_length = len(content)
            print(f"📄 Content loaded: {content_length:,} characters")
            
            # Truncate if too long for testing
            if content_length > 3000:
                content = content[:3000] + "..."
                print(f"📄 Truncated to 3000 chars for testing")
        except Exception as e:
            print(f"❌ Failed to load {file_path.name}: {e}")
            continue
        
        # Run Agent 1 analysis
        try:
            start_time = time.time()
            domain_analysis = await run_domain_analysis(content, detailed=True)
            processing_time = time.time() - start_time
            
            print(f"✅ Agent 1 analysis completed in {processing_time:.2f}s")
            print(f"   Domain signature: {domain_analysis.domain_signature}")
            print(f"   Content confidence: {domain_analysis.content_type_confidence:.3f}")
            
            # Store result for analysis
            results[file_path.name] = {
                'file_info': {
                    'name': file_path.name,
                    'size_bytes': file_path.stat().st_size,
                    'content_length_analyzed': len(content)
                },
                'processing_metrics': {
                    'processing_time_seconds': processing_time,
                    'analysis_timestamp': datetime.now().isoformat()
                },
                'agent1_output': domain_analysis.model_dump() if hasattr(domain_analysis, 'model_dump') else str(domain_analysis)
            }
            
        except Exception as e:
            print(f"❌ Agent 1 analysis failed for {file_path.name}: {e}")
            results[file_path.name] = {
                'file_info': {'name': file_path.name},
                'error': str(e)
            }
            continue
    
    total_time = time.time() - total_start
    print(f"\n⏱️  Total processing time: {total_time:.2f}s")
    
    # 4. Analyze results against UniversalDomainAnalysis schema
    print(f"\n🔍 Step 4: Analyzing Agent 1 output against UniversalDomainAnalysis schema...")
    
    # Import schema for validation
    try:
        from agents.core.universal_models import UniversalDomainAnalysis
        schema_fields = set(UniversalDomainAnalysis.model_fields.keys())
        print(f"📊 UniversalDomainAnalysis schema has {len(schema_fields)} fields")
    except Exception as e:
        print(f"❌ Could not load UniversalDomainAnalysis schema: {e}")
        return False
    
    # Also check centralized schema
    try:
        from agents.core.centralized_agent1_schema import Agent1EssentialOutputSchema
        centralized_fields = set(Agent1EssentialOutputSchema.model_fields.keys())
        print(f"📊 Centralized essential schema has {len(centralized_fields)} fields")
    except Exception as e:
        print(f"❌ Could not load centralized schema: {e}")
        centralized_fields = set()
    
    # Analyze each successful result
    analysis_summary = {
        'files_processed': len([r for r in results.values() if 'agent1_output' in r]),
        'files_failed': len([r for r in results.values() if 'error' in r]),
        'schema_analysis': {},
        'field_coverage': {},
        'centralized_schema_coverage': {}
    }
    
    successful_results = [r for r in results.values() if 'agent1_output' in r]
    
    if successful_results:
        print(f"\n📈 Analyzing {len(successful_results)} successful Agent 1 outputs...")
        
        # Check first successful result for detailed analysis
        sample_output = successful_results[0]['agent1_output']
        if isinstance(sample_output, dict):
            actual_fields = set(sample_output.keys())
            
            # Schema coverage analysis
            missing_from_full_schema = schema_fields - actual_fields
            extra_in_actual = actual_fields - schema_fields
            
            analysis_summary['field_coverage'] = {
                'full_schema_fields_expected': len(schema_fields),
                'actual_output_fields': len(actual_fields),
                'missing_from_output': list(missing_from_full_schema),
                'extra_in_output': list(extra_in_actual),
                'coverage_percentage': (len(actual_fields & schema_fields) / len(schema_fields)) * 100
            }
            
            # Centralized schema coverage
            if centralized_fields:
                centralized_coverage = actual_fields & centralized_fields
                analysis_summary['centralized_schema_coverage'] = {
                    'centralized_fields_expected': len(centralized_fields),
                    'centralized_fields_present': len(centralized_coverage),
                    'centralized_coverage_percentage': (len(centralized_coverage) / len(centralized_fields)) * 100,
                    'missing_centralized_fields': list(centralized_fields - actual_fields)
                }
            
            print(f"✅ Schema analysis complete:")
            print(f"   Full schema coverage: {analysis_summary['field_coverage']['coverage_percentage']:.1f}%")
            if centralized_fields:
                print(f"   Centralized schema coverage: {analysis_summary['centralized_schema_coverage']['centralized_coverage_percentage']:.1f}%")
    
    # 5. Save results locally
    print(f"\n💾 Step 5: Saving results locally...")
    
    output_file = Path('/workspace/azure-maintie-rag/agent1_real_output_analysis.json')
    
    full_report = {
        'test_metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_files_tested': len(real_files),
            'total_processing_time_seconds': total_time,
            'azure_services_used': ['Azure OpenAI'],
            'data_source': str(data_dir)
        },
        'individual_results': results,
        'analysis_summary': analysis_summary
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to: {output_file}")
        print(f"   File size: {output_file.stat().st_size:,} bytes")
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
    
    # 6. Summary and Design Match Analysis
    print(f"\n🎯 DESIGN MATCH ANALYSIS")
    print("=" * 30)
    
    if successful_results:
        print(f"✅ Agent 1 successfully generated UniversalDomainAnalysis output")
        print(f"✅ Processed {len(successful_results)}/{len(real_files)} real Azure AI documentation files")
        print(f"✅ Used real Azure OpenAI service (no mocks)")
        
        coverage = analysis_summary['field_coverage']['coverage_percentage']
        if coverage >= 90:
            print(f"✅ Excellent schema coverage: {coverage:.1f}%")
        elif coverage >= 70:
            print(f"⚠️  Good schema coverage: {coverage:.1f}% (some fields missing)")
        else:
            print(f"❌ Poor schema coverage: {coverage:.1f}% (many fields missing)")
        
        if missing_fields := analysis_summary['field_coverage']['missing_from_output']:
            print(f"⚠️  Missing fields from output: {missing_fields}")
        
        if extra_fields := analysis_summary['field_coverage']['extra_in_output']:
            print(f"ℹ️  Extra fields in output: {extra_fields}")
            
        if centralized_fields:
            centralized_coverage = analysis_summary['centralized_schema_coverage']['centralized_coverage_percentage']
            print(f"✅ Centralized schema coverage: {centralized_coverage:.1f}%")
    else:
        print(f"❌ No successful Agent 1 outputs to analyze")
        return False
    
    print(f"\n📄 Detailed analysis saved to: {output_file}")
    return True

if __name__ == "__main__":
    result = asyncio.run(test_agent1_real_output())
    if result:
        print("\n🎉 Agent 1 real output analysis completed successfully!")
    else:
        print("\n❌ Agent 1 real output analysis failed!")
        sys.exit(1)
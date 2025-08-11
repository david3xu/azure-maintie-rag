#!/usr/bin/env python3
"""
Comprehensive Dataflow Input/Output Analysis
Analyzes data transformation patterns across the entire pipeline
"""

import sys
import json
import asyncio
from pathlib import Path
sys.path.insert(0, '/workspace/azure-maintie-rag')

async def analyze_dataflow_patterns():
    print('🔍 DATAFLOW INPUT/OUTPUT ANALYSIS')
    print('=' * 50)
    
    print('\n1️⃣ DATA SOURCES:')
    
    # Check raw data inputs
    raw_data_path = Path('/workspace/azure-maintie-rag/data/raw/azure-ai-services-language-service_output')
    if raw_data_path.exists():
        files = list(raw_data_path.glob('*.md'))
        print(f'   📁 Raw data files: {len(files)} .md files')
        print(f'   📄 Sample file: {files[0].name}' if files else '   📄 No files found')
        
        # Analyze content format of first file
        if files:
            sample_content = files[0].read_text(encoding='utf-8', errors='ignore')[:500]
            print(f'   📊 Sample content format:')
            print(f'     • Length: {len(sample_content)} chars (first 500)')
            print(f'     • Contains code blocks: {"```" in sample_content}')
            print(f'     • Contains headers: {"#" in sample_content}')
            print(f'     • Contains tables: {"|" in sample_content}')
    
    print('\n2️⃣ AGENT OUTPUT FORMATS:')
    
    # Test Agent 1 output format
    from agents.domain_intelligence.agent import run_domain_analysis
    
    sample_text = 'Azure Machine Learning provides automated ML capabilities for data science workflows.'
    
    try:
        domain_result = await run_domain_analysis(sample_text, detailed=True)
        print(f'   🧠 Agent 1 (Domain Intelligence) Output:')
        print(f'     • Type: {type(domain_result).__name__}')
        print(f'     • Domain signature: {domain_result.domain_signature}')
        print(f'     • Processing config fields: {len(domain_result.processing_config.model_dump())} fields')
        print(f'     • Characteristics fields: {len(domain_result.characteristics.model_dump())} fields')
        print(f'     • Has metadata: {hasattr(domain_result, "analysis_timestamp")}')
        
        # Show actual data structure
        agent1_output = domain_result.model_dump()
        print(f'     • JSON structure: {list(agent1_output.keys())}')
        
    except Exception as e:
        print(f'   ❌ Agent 1 test failed: {str(e)[:80]}...')
    
    print('\n3️⃣ INTERMEDIATE DATA FORMATS:')
    
    # Check processed data directory
    processed_path = Path('/workspace/azure-maintie-rag/data/processed')
    if processed_path.exists():
        processed_files = list(processed_path.rglob('*'))
        print(f'   📁 Processed data files: {len(processed_files)} files')
        for pf in processed_files[:3]:
            print(f'     • {pf.name} ({pf.suffix})')
    else:
        print('   📁 Processed data directory: Not found (data not processed yet)')
    
    print('\n4️⃣ OUTPUT STORAGE PATTERNS:')
    
    # Check session and output locations
    logs_path = Path('/workspace/azure-maintie-rag/logs')
    if logs_path.exists():
        log_files = list(logs_path.glob('*'))
        print(f'   📋 Log files: {len(log_files)} files')
        for lf in log_files[:3]:
            if lf.is_file():
                print(f'     • {lf.name} ({lf.stat().st_size} bytes)')
    
    # Check cache directory
    cache_path = Path('/workspace/azure-maintie-rag/cache')
    if cache_path.exists():
        cache_files = list(cache_path.rglob('*'))
        print(f'   💾 Cache files: {len([f for f in cache_files if f.is_file()])} files')
        for cf in [f for f in cache_files if f.is_file()][:3]:
            print(f'     • {cf.relative_to(cache_path)} ({cf.stat().st_size} bytes)')
    
    print('\n5️⃣ DATA TRANSFORMATION PIPELINE:')
    print('   Raw .md → Domain Analysis → Agent Processing → JSON Output')
    print('   └─ Agent 1: Text → UniversalDomainAnalysis (Pydantic)')
    print('   └─ Agent 2: Content + Domain → ExtractionResult (entities/relationships)')
    print('   └─ Agent 3: Query + Domain → SearchResponse (unified results)')
    
    return True

if __name__ == "__main__":
    asyncio.run(analyze_dataflow_patterns())
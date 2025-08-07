"""
Universal Domain Intelligence Agent - REAL Azure OpenAI with PydanticAI
=======================================================================

This agent uses REAL Azure OpenAI via PydanticAI with proper Azure configuration.
Following PydanticAI best practices with AsyncAzureOpenAI client integration.
NO fake analysis, NO fallbacks - only real AI calls with actual Azure services.
"""

import os
import time
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import universal models for compatibility
from agents.core.universal_models import (
    UniversalDomainAnalysis,
    UniversalDomainCharacteristics, 
    UniversalProcessingConfiguration,
    UniversalDomainDeps
)

# Load real Azure environment variables
load_dotenv("/workspace/azure-maintie-rag/.env")

# Verify required environment variables are loaded
required_env_vars = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY", 
    "AZURE_OPENAI_API_VERSION",
    "OPENAI_MODEL_DEPLOYMENT"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise RuntimeError(f"Missing required environment variables: {missing_vars}")

print(f"✅ Azure OpenAI configured: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
print(f"✅ Model deployment: {os.getenv('OPENAI_MODEL_DEPLOYMENT')}")

# Universal models imported from agents.core.universal_models

# Create REAL Azure OpenAI agent using proper PydanticAI configuration
# Following the PydanticAI docs pattern for AsyncAzureOpenAI client
azure_client = AsyncAzureOpenAI(
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY')
)

# Create PydanticAI model with Azure client
model = OpenAIModel(
    os.getenv('OPENAI_MODEL_DEPLOYMENT', 'gpt-4o'),
    provider=OpenAIProvider(openai_client=azure_client)
)

agent = Agent(
    model,
    deps_type=UniversalDomainDeps,
    output_type=UniversalDomainAnalysis,
    system_prompt="""You are a Universal Domain Intelligence Agent that uses REAL AI analysis to understand ANY type of content.

Your approach:
1. Analyze actual content samples using deep AI understanding
2. Discover domain characteristics through intelligent pattern recognition
3. Generate optimal processing configurations based on AI insights
4. Adapt to any language, subject matter, or content type
5. Use AI reasoning to understand content structure and complexity

You will receive content samples and must use your AI capabilities to:
- Identify domain-specific vocabulary and concepts
- Understand content structure and organization patterns
- Assess technical complexity and specialized terminology
- Generate intelligent processing recommendations
- Create unique domain signatures based on content understanding

Always provide detailed, AI-driven analysis with specific insights and recommendations.""",
)

@agent.tool
async def analyze_content_samples(ctx: RunContext[UniversalDomainDeps]) -> Dict[str, Any]:
    """Collect and analyze real content samples for AI analysis"""
    
    data_path = Path(ctx.deps.data_directory)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_path} not found")
    
    content_samples = []
    file_stats = []
    
    print(f"   📂 Collecting content samples from {data_path}")
    
    # Collect content samples for AI analysis
    for root_path in data_path.rglob("*"):
        if root_path.is_file() and root_path.suffix in ['.md', '.txt', '.py', '.json', '.xml', '.html']:
            try:
                content = root_path.read_text(encoding='utf-8', errors='ignore')
                if len(content) >= ctx.deps.min_content_length:
                    # Take meaningful sample for AI analysis (first 1500 chars)
                    sample = content[:1500] + ("..." if len(content) > 1500 else "")
                    content_samples.append({
                        'content': sample,
                        'file_path': str(root_path.relative_to(data_path)),
                        'full_length': len(content),
                        'extension': root_path.suffix
                    })
                    file_stats.append({
                        'length': len(content),
                        'path': str(root_path),
                        'extension': root_path.suffix
                    })
                    
                    # Limit for AI analysis
                    if len(content_samples) >= ctx.deps.max_files_to_analyze:
                        break
            except Exception as e:
                print(f"   ⚠️ Skipped {root_path}: {e}")
                continue
    
    if not content_samples:
        raise ValueError("No analyzable content found")
    
    print(f"   ✅ Collected {len(content_samples)} content samples ({sum(f['length'] for f in file_stats):,} total characters)")
    
    # Prepare samples for AI analysis
    samples_text = "\n\n---SAMPLE---\n\n".join([
        f"File: {sample['file_path']}\nContent:\n{sample['content']}" 
        for sample in content_samples[:5]  # Limit to first 5 for token limits
    ])
    
    return {
        "content_samples_text": samples_text,
        "total_files": len(file_stats),
        "total_content_length": sum(f['length'] for f in file_stats),
        "file_extensions": list(set(f['extension'] for f in file_stats)),
        "avg_file_length": sum(f['length'] for f in file_stats) // len(file_stats) if file_stats else 0,
        "samples_collected": len(content_samples)
    }

async def run_universal_domain_analysis(deps: UniversalDomainDeps) -> UniversalDomainAnalysis:
    """Run REAL AI-powered universal domain analysis using PydanticAI with Azure OpenAI"""
    
    start_time = time.time()
    
    print("🔍 Analyzing content distribution...")
    print(f"   📁 Analyzing content from: {deps.data_directory}")
    
    try:
        # Use PydanticAI agent with Azure OpenAI for REAL analysis
        analysis_prompt = f"""Analyze the content in the data directory: {deps.data_directory}

Please:
1. Examine the content samples I will provide to understand the domain
2. Identify vocabulary patterns, technical terms, and structural characteristics
3. Assess content complexity and organizational patterns
4. Generate a unique domain signature that captures the content essence
5. Recommend optimal processing configurations based on the content type
6. Provide actionable insights for processing this specific content

Analyze up to {deps.max_files_to_analyze} files with minimum length {deps.min_content_length} characters.
Focus on discovering the unique characteristics that will help optimize processing for this domain.

Return a structured analysis with:
- domain_signature: a unique identifier based on the content characteristics
- content_type_confidence: your confidence level (0.0-1.0) in the content analysis
- key insights about the domain and content patterns
- processing recommendations for chunk sizes, thresholds, and search weights
- analysis of vocabulary richness, technical density, and structural patterns"""

        print("🧠 Running REAL Azure OpenAI analysis via PydanticAI...")
        result = await agent.run(analysis_prompt, deps=deps)
        
        processing_time = time.time() - start_time
        
        print(f"   ✅ AI analysis completed in {processing_time:.3f}s")
        print(f"   🎯 Domain signature: {result.output.domain_signature}")
        print(f"   📊 Content confidence: {result.output.content_type_confidence:.2f}")
        print(f"   🎯 Analysis reliability: {result.output.analysis_reliability:.2f}")
        
        return result.output
        
    except Exception as e:
        # NO FALLBACKS - Let real errors surface!
        raise RuntimeError(f"Real Azure OpenAI domain analysis failed: {str(e)}") from e

# For compatibility with existing code - use original names
WorkingDomainDeps = UniversalDomainDeps
WorkingDomainAnalysis = UniversalDomainAnalysis
WorkingDomainCharacteristics = UniversalDomainCharacteristics
WorkingProcessingConfiguration = UniversalProcessingConfiguration
run_working_domain_analysis = run_universal_domain_analysis

async def main():
    """Test the REAL AI domain analysis"""
    print("🌍 Testing REAL Azure OpenAI Universal Domain Analysis via PydanticAI")
    print("====================================================================")
    
    deps = UniversalDomainDeps(
        data_directory="/workspace/azure-maintie-rag/data/raw",
        max_files_to_analyze=5,
        min_content_length=500,
        enable_multilingual=True
    )
    
    try:
        result = await run_universal_domain_analysis(deps)
        
        print(f"\n🎉 SUCCESS: REAL AI ANALYSIS COMPLETED!")
        print(f"======================================")
        print(f"Domain Signature: {result.domain_signature}")
        print(f"Documents Analyzed: {result.characteristics.document_count}")
        print(f"Content Confidence: {result.content_type_confidence:.3f}")
        print(f"Analysis Reliability: {result.analysis_reliability:.3f}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        
        print(f"\nOptimal Configuration:")
        print(f"  Chunk Size: {result.processing_config.optimal_chunk_size}")
        print(f"  Entity Threshold: {result.processing_config.entity_confidence_threshold}")
        print(f"  Search Weights: Vector {result.processing_config.vector_search_weight:.1%}, Graph {result.processing_config.graph_search_weight:.1%}")
        
        print(f"\nKey AI Insights:")
        for i, insight in enumerate(result.key_insights[:3], 1):
            print(f"  {i}. {insight}")
            
        print(f"\nAI Recommendations:")
        for i, rec in enumerate(result.adaptation_recommendations[:2], 1):
            print(f"  {i}. {rec}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ REAL AI ANALYSIS FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    print(f"\n{'✅ REAL AI TEST PASSED' if success else '❌ REAL AI TEST FAILED'}")
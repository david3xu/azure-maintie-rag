"""
Simplified Domain Intelligence Agent - PydanticAI Best Practices
================================================================

This implementation demonstrates the simplified architecture following PydanticAI best practices:
- Direct agent creation without complexity layers
- Simple dependency injection
- Clean tool definitions
- Focused on core functionality
"""

import os
from typing import Dict, List, Optional
from pathlib import Path
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

# Simple, focused dependencies
class DomainDeps(BaseModel):
    """Simple dependencies for domain intelligence operations"""
    data_directory: str = "/workspace/azure-maintie-rag/data/raw"
    cache_enabled: bool = True

# Clean output model
class DomainAnalysis(BaseModel):
    """Domain analysis output"""
    detected_domain: str
    confidence: float
    file_count: int
    recommendations: List[str]
    processing_time: float

# Lazy agent initialization to avoid import-time requirements
_agent = None

def _create_domain_agent():
    """Internal agent creation with tools"""
    # Get model configuration directly from environment
    model_name = f"openai:{os.getenv('OPENAI_MODEL_DEPLOYMENT', 'gpt-4o')}"
    
    # Create agent with clean, direct configuration
    agent = Agent(
        model_name,
        deps_type=DomainDeps,
        result_type=DomainAnalysis,
        system_prompt="""You are a Domain Intelligence Agent that discovers domains from directory structures.
        
        Your job is to:
        1. Analyze directory structures to identify domains
        2. Count files and assess domain characteristics
        3. Provide configuration recommendations based on domain patterns
        
        Always respond with structured DomainAnalysis output.""",
    )
    
    @agent.tool
    async def discover_domains(ctx: RunContext[DomainDeps]) -> Dict[str, int]:
        """Discover available domains from filesystem"""
        data_path = Path(ctx.deps.data_directory)
        domains = {}
        
        if data_path.exists():
            for subdir in data_path.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("."):
                    domain_name = subdir.name.lower().replace("-", "_")
                    file_count = len(list(subdir.glob("*.md"))) + len(list(subdir.glob("*.txt")))
                    if file_count > 0:
                        domains[domain_name] = file_count
        
        return domains
    
    @agent.tool  
    async def analyze_domain_content(ctx: RunContext[DomainDeps], domain_name: str) -> Dict[str, any]:
        """Analyze content characteristics of a specific domain"""
        domain_path = Path(ctx.deps.data_directory) / domain_name.replace("_", "-")
        
        if not domain_path.exists():
            return {"error": f"Domain path {domain_path} not found"}
        
        files = list(domain_path.glob("*.md")) + list(domain_path.glob("*.txt"))
        
        # Simple content analysis
        total_chars = 0
        avg_file_size = 0
        
        for file_path in files[:5]:  # Sample first 5 files
            try:
                content = file_path.read_text(encoding='utf-8')
                total_chars += len(content)
            except:
                continue
        
        if files:
            avg_file_size = total_chars // min(len(files), 5)
        
        return {
            "total_files": len(files),
            "average_file_size": avg_file_size,
            "domain_type": "technical" if avg_file_size > 10000 else "general",
            "recommended_chunk_size": min(1500, max(500, avg_file_size // 10))
        }
    
    return agent

def create_domain_agent() -> Agent[DomainDeps, DomainAnalysis]:
    """Create domain intelligence agent with PydanticAI best practices"""
    global _agent
    if _agent is None:
        _agent = _create_domain_agent()
    return _agent

# Simple factory function
def get_domain_agent() -> Agent[DomainDeps, DomainAnalysis]:
    """Get domain intelligence agent (no global state)"""
    return create_domain_agent()

# For easy access to tools in testing
def get_agent_tools():
    """Get agent tools for testing (without creating full agent)"""
    return ["discover_domains", "analyze_domain_content"]

# Export simplified interface
__all__ = ["create_domain_agent", "get_domain_agent", "DomainDeps", "DomainAnalysis"]
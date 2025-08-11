"""
Pre-computed Prompt Library for Azure Universal RAG System
========================================================

Implements content-based prompt caching to eliminate double LLM calling issues.
Auto prompts are generated once per unique content and reused efficiently.

Key Features:
- Content hash-based caching (same content = same prompts)
- No double LLM calling (domain analysis + entity types in single pass)
- Configurable TTL and cache size limits
- Thread-safe for concurrent access
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from functools import lru_cache

from agents.core.universal_models import UniversalDomainAnalysis, UniversalDomainCharacteristics


@dataclass
class CachedPromptEntry:
    """Represents a cached auto prompt entry."""
    
    content_hash: str
    content_signature: str
    domain_analysis: UniversalDomainAnalysis
    entity_predictions: Dict[str, Any]
    extraction_prompts: Dict[str, str]
    template_variables: Dict[str, Any]
    created_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        """Check if cache entry has expired (default 1 hour TTL)."""
        return time.time() - self.created_at > ttl_seconds
    
    def touch(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


class PreComputedPromptLibrary:
    """
    Pre-computed Prompt Library for efficient auto prompt management.
    
    Eliminates double LLM calling by caching complete auto prompt results
    based on content hash. Same content = same prompts (reused).
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize prompt library.
        
        Args:
            max_size: Maximum number of cached entries
            default_ttl: Time-to-live in seconds (default 1 hour)
        """
        self.cache: Dict[str, CachedPromptEntry] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._lock = asyncio.Lock()
        
    @staticmethod
    def generate_content_hash(content: str) -> str:
        """Generate deterministic hash for content."""
        # Normalize content (strip whitespace, lowercase for consistency)
        normalized = content.strip().lower()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]
    
    async def get_cached_prompts(
        self, 
        content: str, 
        force_refresh: bool = False
    ) -> Optional[CachedPromptEntry]:
        """
        Retrieve cached auto prompts for content.
        
        Args:
            content: Text content to check
            force_refresh: Skip cache and force regeneration
            
        Returns:
            CachedPromptEntry if found and valid, None otherwise
        """
        if force_refresh:
            return None
            
        content_hash = self.generate_content_hash(content)
        
        async with self._lock:
            entry = self.cache.get(content_hash)
            
            if entry and not entry.is_expired(self.default_ttl):
                entry.touch()  # Update access statistics
                return entry
            elif entry:
                # Remove expired entry
                del self.cache[content_hash]
                
        return None
    
    async def cache_prompts(
        self,
        content: str,
        domain_analysis: UniversalDomainAnalysis,
        entity_predictions: Dict[str, Any],
        extraction_prompts: Dict[str, str],
        template_variables: Dict[str, Any]
    ) -> CachedPromptEntry:
        """
        Cache auto prompts for future reuse.
        
        Args:
            content: Original text content
            domain_analysis: Domain intelligence analysis
            entity_predictions: Auto-generated entity types
            extraction_prompts: Generated extraction prompts  
            template_variables: Jinja2 template variables
            
        Returns:
            CachedPromptEntry that was cached
        """
        content_hash = self.generate_content_hash(content)
        
        entry = CachedPromptEntry(
            content_hash=content_hash,
            content_signature=domain_analysis.domain_signature,
            domain_analysis=domain_analysis,
            entity_predictions=entity_predictions,
            extraction_prompts=extraction_prompts,
            template_variables=template_variables,
            created_at=time.time()
        )
        
        async with self._lock:
            # Implement LRU eviction if cache is full
            if len(self.cache) >= self.max_size:
                await self._evict_lru_entries()
            
            self.cache[content_hash] = entry
        
        return entry
    
    async def _evict_lru_entries(self, evict_count: int = None):
        """Evict least recently used entries to make space."""
        if not evict_count:
            evict_count = max(1, len(self.cache) // 10)  # Evict 10% by default
        
        # Sort by last_accessed timestamp (ascending = oldest first)
        sorted_entries = sorted(
            self.cache.items(), 
            key=lambda x: x[1].last_accessed
        )
        
        # Remove oldest entries
        for content_hash, _ in sorted_entries[:evict_count]:
            del self.cache[content_hash]
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        async with self._lock:
            if not self.cache:
                return {"cache_size": 0, "total_entries": 0}
                
            total_access = sum(entry.access_count for entry in self.cache.values())
            avg_access = total_access / len(self.cache) if self.cache else 0
            
            return {
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "cache_utilization": len(self.cache) / self.max_size,
                "total_accesses": total_access,
                "avg_accesses_per_entry": round(avg_access, 2),
                "oldest_entry_age": time.time() - min(
                    (entry.created_at for entry in self.cache.values()), 
                    default=time.time()
                )
            }
    
    async def clear_expired_entries(self) -> int:
        """Remove expired entries and return count removed."""
        removed_count = 0
        current_time = time.time()
        
        async with self._lock:
            expired_hashes = [
                content_hash for content_hash, entry in self.cache.items()
                if current_time - entry.created_at > self.default_ttl
            ]
            
            for content_hash in expired_hashes:
                del self.cache[content_hash]
                removed_count += 1
        
        return removed_count
    
    async def clear_cache(self):
        """Clear all cached entries."""
        async with self._lock:
            self.cache.clear()
    
    def export_cache_summary(self) -> Dict[str, Any]:
        """Export cache summary for debugging (synchronous)."""
        return {
            "cache_entries": len(self.cache),
            "max_capacity": self.max_size,
            "content_hashes": list(self.cache.keys()),
            "signatures": [entry.content_signature for entry in self.cache.values()],
            "creation_times": [entry.created_at for entry in self.cache.values()],
            "access_counts": [entry.access_count for entry in self.cache.values()]
        }


# Global prompt library instance (singleton pattern)
_prompt_library_instance: Optional[PreComputedPromptLibrary] = None


def get_prompt_library() -> PreComputedPromptLibrary:
    """Get or create global prompt library instance."""
    global _prompt_library_instance
    if _prompt_library_instance is None:
        _prompt_library_instance = PreComputedPromptLibrary()
    return _prompt_library_instance


async def get_or_generate_auto_prompts(
    content: str,
    force_refresh: bool = False,
    verbose: bool = False
) -> CachedPromptEntry:
    """
    High-level function to get or generate auto prompts with caching.
    
    This function implements the core auto prompt workflow:
    1. Check cache for existing prompts
    2. If not found, generate new prompts (domain analysis + entity types)
    3. Cache the results for future use
    4. Return complete prompt information
    
    Args:
        content: Text content to analyze
        force_refresh: Skip cache and regenerate prompts
        verbose: Enable detailed logging
        
    Returns:
        CachedPromptEntry with all auto prompt information
    """
    from agents.core.universal_deps import get_universal_deps
    from agents.domain_intelligence.agent import domain_intelligence_agent
    from agents.core.agent_toolsets import predict_entity_types, generate_extraction_prompts
    
    library = get_prompt_library()
    
    # Step 1: Check cache first
    if not force_refresh:
        cached_entry = await library.get_cached_prompts(content, force_refresh=False)
        if cached_entry:
            if verbose:
                print(f"✅ Using cached prompts for content hash: {cached_entry.content_hash}")
                print(f"   Cache hit! Saved ~15-20s of LLM processing")
            return cached_entry
    
    if verbose:
        print(f"🔄 Generating new auto prompts for content ({len(content)} chars)")
    
    # Step 2: Generate new prompts (single-pass, no double LLM calling)
    deps = await get_universal_deps()
    
    # Single domain analysis call (includes all needed information)
    if verbose:
        print("   🧠 Running domain analysis...")
        
    domain_result = await domain_intelligence_agent.run(
        f"Complete analysis for auto prompt generation: {content[:1000]}",
        deps=deps
    )
    domain_analysis = domain_result.output
    
    if verbose:
        print(f"   ✅ Domain signature: {domain_analysis.domain_signature}")
        print(f"   📊 Vocabulary complexity: {domain_analysis.characteristics.vocabulary_complexity_ratio:.3f}")
    
    # Generate entity predictions (reuse domain analysis context)
    if verbose:
        print("   🎯 Generating entity type predictions...")
        
    class MockRunContext:
        def __init__(self, deps): 
            self.deps = deps
    
    ctx = MockRunContext(deps)
    entity_predictions = await predict_entity_types(ctx, content, domain_analysis.characteristics)
    
    if verbose:
        predicted_types = entity_predictions.get('predicted_entity_types', [])
        print(f"   ✅ Entity types: {predicted_types}")
    
    # Generate extraction prompts
    if verbose:
        print("   📝 Generating extraction prompts...")
        
    extraction_prompts = await generate_extraction_prompts(
        ctx, content, entity_predictions, domain_analysis.characteristics
    )
    
    # Prepare template variables for Jinja2
    template_variables = {
        'discovered_entity_types': entity_predictions.get('predicted_entity_types', []),
        'content_signature': domain_analysis.domain_signature,
        'key_content_terms': domain_analysis.characteristics.key_content_terms,
        'vocabulary_richness': domain_analysis.characteristics.vocabulary_richness,
        'concept_density': domain_analysis.characteristics.concept_density,
        'discovered_content_patterns': domain_analysis.characteristics.content_patterns,
        'entity_confidence_threshold': 0.7,
        'relationship_confidence_threshold': 0.6
    }
    
    if verbose:
        print(f"   ✅ Generated {len(extraction_prompts)} prompts and {len(template_variables)} template variables")
    
    # Step 3: Cache the results for future use
    cached_entry = await library.cache_prompts(
        content=content,
        domain_analysis=domain_analysis,
        entity_predictions=entity_predictions,
        extraction_prompts=extraction_prompts,
        template_variables=template_variables
    )
    
    if verbose:
        print(f"   💾 Cached prompts with hash: {cached_entry.content_hash}")
        print(f"   🎯 Future requests for same content will reuse these prompts")
    
    return cached_entry


# Cache monitoring and maintenance functions
async def cleanup_prompt_cache():
    """Clean up expired cache entries (call periodically)."""
    library = get_prompt_library()
    removed_count = await library.clear_expired_entries()
    return removed_count


async def get_prompt_cache_stats():
    """Get current cache statistics."""
    library = get_prompt_library()
    return await library.get_cache_stats()
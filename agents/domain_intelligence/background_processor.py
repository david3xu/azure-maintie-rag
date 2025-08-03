"""
Domain Background Processor - Startup processing for optimal runtime performance

This service processes all domain documents at system startup, extracting patterns,
generating configurations, and caching everything for lightning-fast runtime queries.

Key features:
- Process all domains in parallel during startup
- Extract and cache domain signatures with statistical patterns
- Pre-generate infrastructure and ML configurations
- Build pattern indexes for O(1) query matching
- Achieve >95% cache hit rates for domain detection
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from .domain_analyzer import DomainAnalyzer as ContentAnalyzer, ContentAnalysis
from .pattern_engine import PatternEngine as PatternExtractor, ExtractedPatterns
from .domain_analyzer import DomainAnalyzer as DomainClassifier, DomainClassification
from .config_generator import ConfigGenerator, DomainConfig
from ..core.cache_manager import UnifiedCacheManager as DomainCache

logger = logging.getLogger(__name__)


@dataclass
class DomainSignature:
    """Domain signature containing all processed domain information"""
    domain: str
    patterns: ExtractedPatterns
    config: DomainConfig
    content_analysis: ContentAnalysis
    classification: DomainClassification
    signature_hash: str
    processing_timestamp: float
    cache_key: str


class BackgroundProcessingStats:
    """Statistics for background processing performance tracking"""
    
    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.domains_processed: int = 0
        self.files_processed: int = 0
        self.patterns_extracted: int = 0
        self.configurations_generated: int = 0
        self.cache_entries_created: int = 0
        self.processing_errors: List[str] = []
    
    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time if self.end_time > 0 else 0.0
    
    @property
    def files_per_second(self) -> float:
        return self.files_processed / self.total_time if self.total_time > 0 else 0.0
    
    def to_dict(self) -> Dict:
        return {
            "total_time": self.total_time,
            "domains_processed": self.domains_processed,
            "files_processed": self.files_processed,
            "patterns_extracted": self.patterns_extracted,
            "configurations_generated": self.configurations_generated,
            "cache_entries_created": self.cache_entries_created,
            "files_per_second": self.files_per_second,
            "processing_errors": len(self.processing_errors),
            "success_rate": (self.files_processed - len(self.processing_errors)) / max(1, self.files_processed)
        }


class DomainBackgroundProcessor:
    """
    Background processor that handles all heavy domain intelligence work at startup
    """
    
    def __init__(self, data_dir: str = "/workspace/azure-maintie-rag/data/raw"):
        self.data_dir = Path(data_dir)
        self.stats = BackgroundProcessingStats()
        
        # Initialize domain processing components
        self.content_analyzer = ContentAnalyzer()
        self.pattern_extractor = PatternExtractor()
        self.domain_classifier = DomainClassifier()
        self.config_generator = ConfigGenerator()
        self.domain_cache = DomainCache()
        
        # Thread pool for I/O operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Domain Background Processor initialized")
    
    async def process_all_domains_on_startup(self) -> BackgroundProcessingStats:
        """
        Main background processing entry point - process all domains at startup
        """
        logger.info("🚀 Starting background domain processing...")
        self.stats.start_time = time.time()
        
        try:
            # 1. Discover available domains from filesystem
            domains = await self._discover_domains()
            logger.info(f"📁 Discovered {len(domains)} domains: {domains}")
            
            # 2. Process all domains in parallel for optimal performance
            domain_tasks = []
            for domain in domains:
                task = self._process_domain_completely(domain)
                domain_tasks.append(task)
            
            # Execute all domain processing concurrently
            domain_results = await asyncio.gather(*domain_tasks, return_exceptions=True)
            
            # 3. Process results and collect statistics
            for i, result in enumerate(domain_results):
                if isinstance(result, Exception):
                    error_msg = f"Domain {domains[i]} processing failed: {result}"
                    logger.error(error_msg)
                    self.stats.processing_errors.append(error_msg)
                else:
                    self.stats.domains_processed += 1
            
            # 4. Build global pattern indexes for fast query matching
            await self._build_global_pattern_indexes()
            
            # 5. Optimize cache for runtime performance
            await self._optimize_cache_for_runtime()
            
            self.stats.end_time = time.time()
            
            # Log completion statistics
            stats_dict = self.stats.to_dict()
            logger.info(f"✅ Background processing complete in {stats_dict['total_time']:.2f}s")
            logger.info(f"📊 Processed {stats_dict['domains_processed']} domains, "
                       f"{stats_dict['files_processed']} files, "
                       f"extracted {stats_dict['patterns_extracted']} patterns")
            logger.info(f"⚡ Processing rate: {stats_dict['files_per_second']:.1f} files/sec")
            logger.info(f"🎯 Success rate: {stats_dict['success_rate']*100:.1f}%")
            
            return self.stats
            
        except Exception as e:
            self.stats.end_time = time.time()
            logger.error(f"❌ Background processing failed: {e}")
            raise
    
    async def _discover_domains(self) -> List[str]:
        """Discover available domains from filesystem structure"""
        domains = []
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory not found: {self.data_dir}")
            return domains
        
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                # Check if directory contains processable files
                has_files = any(
                    subdir.glob("**/*.md") or 
                    subdir.glob("**/*.txt")
                )
                if has_files:
                    domains.append(subdir.name)
        
        return sorted(domains)
    
    async def _process_domain_completely(self, domain: str) -> Dict:
        """
        Process all files for a domain and cache everything for runtime performance
        """
        domain_start = time.time()
        logger.info(f"🔄 Processing domain: {domain}")
        
        try:
            domain_dir = self.data_dir / domain
            
            # 1. Find all processable files in domain directory
            files = list(domain_dir.glob("**/*.md")) + list(domain_dir.glob("**/*.txt"))
            if not files:
                logger.warning(f"No files found in domain directory: {domain_dir}")
                return {"domain": domain, "files_processed": 0, "patterns_extracted": 0}
            
            logger.info(f"📄 Found {len(files)} files in domain {domain}")
            
            # 2. Process all files in parallel
            file_tasks = []
            for file_path in files:
                task = self._process_file_for_domain(domain, file_path)
                file_tasks.append(task)
            
            file_results = await asyncio.gather(*file_tasks, return_exceptions=True)
            
            # 3. Collect all patterns from processed files
            all_patterns = []
            files_processed = 0
            
            for i, result in enumerate(file_results):
                if isinstance(result, Exception):
                    error_msg = f"File {files[i]} processing failed: {result}"
                    logger.error(error_msg)
                    self.stats.processing_errors.append(error_msg)
                else:
                    all_patterns.append(result)
                    files_processed += 1
            
            if not all_patterns:
                logger.warning(f"No patterns extracted for domain {domain}")
                return {"domain": domain, "files_processed": 0, "patterns_extracted": 0}
            
            # 4. Create consolidated domain signature from all patterns
            domain_signature = await self._create_consolidated_domain_signature(domain, all_patterns)
            
            # 5. Generate complete domain configuration
            domain_config = self.config_generator.generate_complete_config(domain, all_patterns[0])
            
            # 6. Cache everything for instant runtime access
            self.domain_cache.set_domain_signature(domain, domain_signature)
            self.domain_cache.set_domain_config(domain, domain_config)
            
            # 7. Update statistics
            total_patterns = sum(
                len(patterns.entity_patterns) + len(patterns.action_patterns) + len(patterns.relationship_patterns)
                for patterns in all_patterns
            )
            
            self.stats.files_processed += files_processed
            self.stats.patterns_extracted += total_patterns
            self.stats.configurations_generated += 1
            self.stats.cache_entries_created += 2  # signature + config
            
            domain_time = time.time() - domain_start
            logger.info(f"✅ Domain {domain} processed in {domain_time:.2f}s: "
                       f"{files_processed} files, {total_patterns} patterns")
            
            return {
                "domain": domain,
                "files_processed": files_processed,
                "patterns_extracted": total_patterns,
                "processing_time": domain_time
            }
            
        except Exception as e:
            error_msg = f"Domain {domain} processing failed: {e}"
            logger.error(error_msg)
            self.stats.processing_errors.append(error_msg)
            raise
    
    async def _process_file_for_domain(self, domain: str, file_path: Path) -> ExtractedPatterns:
        """Process a single file and extract patterns"""
        
        try:
            # 1. Analyze file content
            content_analysis = self.content_analyzer.analyze_raw_content(file_path)
            
            # 2. Classify domain (validate against expected domain)
            classification = self.domain_classifier.classify_content_domain(content_analysis, domain)
            
            # 3. Extract statistical patterns
            patterns = self.pattern_extractor.extract_domain_patterns(
                classification.domain, 
                content_analysis, 
                classification.confidence
            )
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            raise
    
    async def _create_consolidated_domain_signature(
        self, 
        domain: str, 
        all_patterns: List[ExtractedPatterns]
    ) -> DomainSignature:
        """Create a consolidated domain signature from all extracted patterns"""
        
        # Aggregate all patterns
        all_entities = []
        all_actions = []
        all_relationships = []
        total_word_count = 0
        
        for patterns in all_patterns:
            all_entities.extend(patterns.entity_patterns)
            all_actions.extend(patterns.action_patterns)
            all_relationships.extend(patterns.relationship_patterns)
            total_word_count += patterns.source_word_count
        
        # Merge similar patterns to avoid duplicates
        merged_entities = self.pattern_extractor.merge_similar_patterns(all_entities)
        merged_actions = self.pattern_extractor.merge_similar_patterns(all_actions)
        merged_relationships = self.pattern_extractor.merge_similar_patterns(all_relationships)
        
        # Calculate overall confidence based on pattern quality
        if merged_entities:
            avg_confidence = sum(p.confidence for p in merged_entities) / len(merged_entities)
        else:
            avg_confidence = 0.3
        
        # Create primary concepts from high-confidence entities
        primary_concepts = [
            p.pattern_text for p in merged_entities 
            if p.confidence > 0.8
        ][:5]  # Top 5 concepts
        
        # Convert patterns to dictionaries for serialization
        entity_dicts = [asdict(p) for p in merged_entities]
        action_dicts = [asdict(p) for p in merged_actions]
        relationship_dicts = [asdict(p) for p in merged_relationships]
        
        return DomainSignature(
            domain_name=domain,
            primary_concepts=primary_concepts,
            entity_patterns=entity_dicts,
            action_patterns=action_dicts,
            relationship_patterns=relationship_dicts,
            confidence_score=avg_confidence,
            sample_size=len(all_patterns),  # Number of files processed
            total_word_count=total_word_count
        )
    
    async def _build_global_pattern_indexes(self):
        """Build global pattern indexes for fast query matching"""
        logger.info("🔧 Building global pattern indexes...")
        
        # The domain cache already builds pattern indexes as signatures are added
        # This method can be extended for additional indexing optimizations
        
        index_stats = {
            "pattern_index_size": len(self.domain_cache._pattern_index),
            "total_domains": len([k for k in self.domain_cache._memory_cache.keys() if k.startswith("signature_")])
        }
        
        logger.info(f"✅ Pattern indexes built: {index_stats['pattern_index_size']} patterns, "
                   f"{index_stats['total_domains']} domains")
    
    async def _optimize_cache_for_runtime(self):
        """Optimize cache structure for runtime performance"""
        logger.info("🚀 Optimizing cache for runtime performance...")
        
        # Clean up any expired entries
        expired_count = self.domain_cache.clear_expired_entries()
        
        # Get cache statistics
        cache_stats = self.domain_cache.get_cache_stats()
        
        logger.info(f"✅ Cache optimized: {cache_stats['active_entries']} active entries, "
                   f"{expired_count} expired entries cleaned, "
                   f"{cache_stats['pattern_index_size']} pattern indexes ready")
    
    async def get_processing_status(self) -> Dict:
        """Get current background processing status and statistics"""
        return {
            "is_processing": self.stats.start_time > 0 and self.stats.end_time == 0,
            "processing_stats": self.stats.to_dict(),
            "cache_stats": self.domain_cache.get_cache_stats()
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        logger.info("Background processor cleanup complete")


# Global background processor instance
_global_processor: Optional[DomainBackgroundProcessor] = None


async def get_background_processor() -> DomainBackgroundProcessor:
    """Get or create global background processor instance"""
    global _global_processor
    
    if _global_processor is None:
        _global_processor = DomainBackgroundProcessor()
    
    return _global_processor


async def run_startup_background_processing() -> BackgroundProcessingStats:
    """
    Convenience function to run complete background processing at startup
    """
    processor = await get_background_processor()
    return await processor.process_all_domains_on_startup()
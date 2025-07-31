"""
Memory Manager - Intelligent memory management for agent reasoning and learning.
Provides hierarchical memory with automatic summarization and pattern extraction.
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, OrderedDict
import logging
import hashlib

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory stored by the memory manager"""
    WORKING = "working"          # Short-term working memory
    EPISODIC = "episodic"        # Episode-based memories (conversations, interactions)
    SEMANTIC = "semantic"        # Factual knowledge and patterns
    PROCEDURAL = "procedural"    # Learned procedures and skills
    META = "meta"                # Meta-knowledge about reasoning and performance


class MemoryPriority(Enum):
    """Memory priority levels for retention decisions"""
    CRITICAL = "critical"        # Never expires, highest importance
    HIGH = "high"               # Long retention, frequent access
    MEDIUM = "medium"           # Standard retention
    LOW = "low"                 # Short retention, infrequent access
    TEMPORARY = "temporary"     # Very short retention


@dataclass
class MemoryEntry:
    """Individual memory entry with metadata and access patterns"""
    memory_id: str
    memory_type: MemoryType
    priority: MemoryPriority
    content: Dict[str, Any]
    created_at: float
    last_accessed: float
    access_count: int = 0
    confidence: float = 1.0
    source: str = "agent"
    expiry_time: Optional[float] = None
    embedding: Optional[List[float]] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def access(self) -> None:
        """Record memory access and update patterns"""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def is_expired(self) -> bool:
        """Check if memory has expired"""
        if self.expiry_time is None:
            return False
        return time.time() > self.expiry_time
    
    def calculate_importance(self) -> float:
        """Calculate memory importance score for retention decisions"""
        # Base importance from priority
        priority_weights = {
            MemoryPriority.CRITICAL: 1.0,
            MemoryPriority.HIGH: 0.8,
            MemoryPriority.MEDIUM: 0.6,
            MemoryPriority.LOW: 0.4,
            MemoryPriority.TEMPORARY: 0.2
        }
        
        base_score = priority_weights[self.priority]
        
        # Boost from recency and frequency
        age_factor = max(0.1, 1.0 - (time.time() - self.created_at) / (7 * 24 * 3600))  # 7 days
        access_factor = min(1.0, self.access_count / 10.0)  # Cap at 10 accesses
        confidence_factor = self.confidence
        
        return base_score * (0.4 + 0.3 * age_factor + 0.2 * access_factor + 0.1 * confidence_factor)


@dataclass
class MemoryCluster:
    """Cluster of related memories for efficient organization"""
    cluster_id: str
    theme: str
    memories: List[str]  # Memory IDs
    centroid_embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    coherence_score: float = 0.0
    
    def add_memory(self, memory_id: str) -> None:
        """Add memory to cluster"""
        if memory_id not in self.memories:
            self.memories.append(memory_id)
            self.last_updated = time.time()


class IntelligentMemoryManager:
    """
    Intelligent memory management system for agents with hierarchical organization.
    
    Features:
    - Multi-level memory hierarchy (working, episodic, semantic, procedural, meta)
    - Automatic summarization and consolidation
    - Pattern extraction and learning
    - Memory importance scoring and retention
    - Efficient retrieval with semantic similarity
    - Memory clustering for organization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize memory manager with data-driven configuration.
        
        Args:
            config: Memory manager configuration
                - max_working_memory: Maximum working memory entries
                - max_episodic_memory: Maximum episodic memory entries  
                - consolidation_interval: How often to consolidate memories
                - similarity_threshold: Threshold for memory clustering
                - retention_policy: Memory retention configuration
                - enable_summarization: Enable automatic summarization
        """
        self.config = config
        self.max_working_memory = config.get("max_working_memory", 100)
        self.max_episodic_memory = config.get("max_episodic_memory", 1000)
        self.consolidation_interval = config.get("consolidation_interval", 3600)  # 1 hour
        self.similarity_threshold = config.get("similarity_threshold", 0.8)
        self.retention_policy = config.get("retention_policy", {
            MemoryPriority.CRITICAL.value: None,  # Never expire
            MemoryPriority.HIGH.value: 30 * 24 * 3600,  # 30 days
            MemoryPriority.MEDIUM.value: 7 * 24 * 3600,  # 7 days
            MemoryPriority.LOW.value: 24 * 3600,  # 1 day
            MemoryPriority.TEMPORARY.value: 3600  # 1 hour
        })
        self.enable_summarization = config.get("enable_summarization", True)
        
        # Memory storage organized by type
        self.memories: Dict[MemoryType, Dict[str, MemoryEntry]] = {
            memory_type: OrderedDict() for memory_type in MemoryType
        }
        
        # Memory clusters for organization
        self.clusters: Dict[str, MemoryCluster] = {}
        
        # Memory patterns and insights
        self.patterns: Dict[str, Any] = {}
        
        # Performance tracking
        self.metrics = {
            "memories_stored": 0,
            "memories_retrieved": 0,
            "memories_consolidated": 0,
            "memories_expired": 0,
            "consolidation_runs": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._consolidation_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Fast retrieval caches
        self._content_cache: Dict[str, str] = {}
        self._embedding_cache: Dict[str, List[float]] = {}
    
    async def initialize(self) -> None:
        """Initialize memory manager and start background tasks"""
        self.logger.info("Initializing intelligent memory manager")
        
        # Start consolidation task
        self._consolidation_task = asyncio.create_task(self._periodic_consolidation())
        
        self.logger.info(f"Memory manager initialized with {len(MemoryType)} memory types")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown memory manager"""
        if self._consolidation_task:
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Memory manager shutdown complete")
    
    async def store_memory(
        self,
        memory_type: MemoryType,
        content: Dict[str, Any],
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        source: str = "agent",
        confidence: float = 1.0,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a new memory entry with automatic organization.
        
        Args:
            memory_type: Type of memory to store
            content: Memory content
            priority: Memory priority level
            source: Source of the memory
            confidence: Confidence in memory accuracy
            tags: Optional tags for categorization
            metadata: Optional metadata
            
        Returns:
            Memory ID for the stored memory
        """
        async with self._lock:
            # Generate memory ID
            content_hash = hashlib.md5(json.dumps(content, sort_keys=True).encode()).hexdigest()
            memory_id = f"{memory_type.value}_{int(time.time())}_{content_hash[:8]}"
            
            # Calculate expiry time
            expiry_time = None
            if priority in self.retention_policy and self.retention_policy[priority.value]:
                expiry_time = time.time() + self.retention_policy[priority.value]
            
            # Create memory entry
            memory = MemoryEntry(
                memory_id=memory_id,
                memory_type=memory_type,
                priority=priority,
                content=content,
                created_at=time.time(),
                last_accessed=time.time(),
                confidence=confidence,
                source=source,
                expiry_time=expiry_time,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # Store memory
            self.memories[memory_type][memory_id] = memory
            self.metrics["memories_stored"] += 1
            
            # Manage memory limits
            await self._enforce_memory_limits(memory_type)
            
            # Update clusters if enabled
            if memory_type in [MemoryType.EPISODIC, MemoryType.SEMANTIC]:
                await self._update_memory_clusters(memory_id, memory)
            
            self.logger.debug(f"Stored {memory_type.value} memory: {memory_id}")
            return memory_id
    
    async def retrieve_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[MemoryEntry]:
        """
        Retrieve memories based on type, content similarity, or tags.
        
        Args:
            memory_type: Optional memory type filter
            query: Optional query for content similarity
            tags: Optional tag filters
            limit: Maximum number of memories to retrieve
            similarity_threshold: Minimum similarity for query matching
            
        Returns:
            List of matching memory entries
        """
        self.metrics["memories_retrieved"] += 1
        
        # Determine memory types to search
        types_to_search = [memory_type] if memory_type else list(MemoryType)
        
        matching_memories = []
        
        for mem_type in types_to_search:
            for memory in self.memories[mem_type].values():
                # Skip expired memories
                if memory.is_expired():
                    continue
                
                # Check tag filters
                if tags and not any(tag in memory.tags for tag in tags):
                    continue
                
                # Check content similarity if query provided
                if query:
                    similarity = await self._calculate_content_similarity(query, memory.content)
                    if similarity < similarity_threshold:
                        continue
                    memory.metadata["similarity_score"] = similarity
                
                # Access the memory (updates access patterns)
                memory.access()
                matching_memories.append(memory)
        
        # Sort by importance and recency
        matching_memories.sort(
            key=lambda m: (
                m.calculate_importance(),
                m.metadata.get("similarity_score", 0.0),
                m.last_accessed
            ),
            reverse=True
        )
        
        return matching_memories[:limit]
    
    async def get_working_memory(self) -> List[MemoryEntry]:
        """Get current working memory contents"""
        working_memories = list(self.memories[MemoryType.WORKING].values())
        
        # Sort by recency and importance
        working_memories.sort(
            key=lambda m: (m.calculate_importance(), m.last_accessed),
            reverse=True
        )
        
        return working_memories
    
    async def consolidate_episodic_memory(self, session_id: str) -> Dict[str, Any]:
        """
        Consolidate episodic memories from a session into semantic memory.
        
        Args:
            session_id: Session to consolidate
            
        Returns:
            Consolidation results and statistics
        """
        async with self._lock:
            # Find episodic memories for session
            session_memories = [
                memory for memory in self.memories[MemoryType.EPISODIC].values()
                if memory.metadata.get("session_id") == session_id
            ]
            
            if len(session_memories) < 2:
                return {"consolidated_memories": 0, "patterns_extracted": 0}
            
            # Extract patterns and common themes
            patterns = await self._extract_memory_patterns(session_memories)
            
            # Create consolidated semantic memories
            consolidated_count = 0
            for pattern_name, pattern_data in patterns.items():
                semantic_content = {
                    "pattern_type": pattern_name,
                    "pattern_data": pattern_data,
                    "source_memories": [m.memory_id for m in session_memories],
                    "consolidation_timestamp": time.time(),
                    "session_id": session_id
                }
                
                await self.store_memory(
                    memory_type=MemoryType.SEMANTIC,
                    content=semantic_content,
                    priority=MemoryPriority.HIGH,
                    source="consolidation",
                    confidence=pattern_data.get("confidence", 0.8),
                    tags=["consolidated", "pattern", pattern_name],
                    metadata={"consolidation_source": session_id}
                )
                consolidated_count += 1
            
            self.metrics["memories_consolidated"] += consolidated_count
            
            # Optionally reduce priority of source episodic memories
            for memory in session_memories:
                if memory.priority != MemoryPriority.CRITICAL:
                    memory.priority = MemoryPriority.LOW
            
            results = {
                "consolidated_memories": consolidated_count,
                "patterns_extracted": len(patterns),
                "source_memories_processed": len(session_memories)
            }
            
            self.logger.info(f"Consolidated session {session_id}: {results}")
            return results
    
    async def extract_learned_patterns(self) -> Dict[str, Any]:
        """
        Extract learned patterns from semantic and procedural memory.
        
        Returns:
            Dictionary of extracted patterns and insights
        """
        patterns = {}
        
        # Analyze semantic memory for knowledge patterns
        semantic_memories = list(self.memories[MemoryType.SEMANTIC].values())
        if semantic_memories:
            patterns["semantic_patterns"] = await self._analyze_semantic_patterns(semantic_memories)
        
        # Analyze procedural memory for skill patterns
        procedural_memories = list(self.memories[MemoryType.PROCEDURAL].values())
        if procedural_memories:
            patterns["procedural_patterns"] = await self._analyze_procedural_patterns(procedural_memories)
        
        # Analyze meta memory for reasoning patterns
        meta_memories = list(self.memories[MemoryType.META].values())
        if meta_memories:
            patterns["meta_patterns"] = await self._analyze_meta_patterns(meta_memories)
        
        # Update global patterns
        self.patterns.update(patterns)
        
        return patterns
    
    async def cleanup_expired_memories(self) -> Dict[str, int]:
        """
        Clean up expired memories and return statistics.
        
        Returns:
            Cleanup statistics
        """
        async with self._lock:
            cleanup_stats = {"total_removed": 0, "by_type": {}}
            
            for memory_type, memories in self.memories.items():
                expired_ids = [
                    memory_id for memory_id, memory in memories.items()
                    if memory.is_expired()
                ]
                
                for memory_id in expired_ids:
                    del memories[memory_id]
                
                cleanup_stats["by_type"][memory_type.value] = len(expired_ids)
                cleanup_stats["total_removed"] += len(expired_ids)
            
            self.metrics["memories_expired"] += cleanup_stats["total_removed"]
            
            # Clean up orphaned clusters
            await self._cleanup_empty_clusters()
            
            if cleanup_stats["total_removed"] > 0:
                self.logger.info(f"Memory cleanup: {cleanup_stats}")
            
            return cleanup_stats
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory manager statistics"""
        memory_counts = {
            memory_type.value: len(memories)
            for memory_type, memories in self.memories.items()
        }
        
        total_memories = sum(memory_counts.values())
        
        # Calculate memory efficiency metrics
        working_memory_usage = len(self.memories[MemoryType.WORKING]) / max(1, self.max_working_memory)
        episodic_memory_usage = len(self.memories[MemoryType.EPISODIC]) / max(1, self.max_episodic_memory)
        
        return {
            "memory_counts": memory_counts,
            "total_memories": total_memories,
            "active_clusters": len(self.clusters),
            "patterns_learned": len(self.patterns),
            "working_memory_usage": working_memory_usage,
            "episodic_memory_usage": episodic_memory_usage,
            "performance_metrics": self.metrics,
            "cache_hit_rate": (
                self.metrics["cache_hits"] / 
                max(1, self.metrics["cache_hits"] + self.metrics["cache_misses"])
            )
        }
    
    # Private implementation methods
    
    async def _enforce_memory_limits(self, memory_type: MemoryType) -> None:
        """Enforce memory limits by removing least important memories"""
        memories = self.memories[memory_type]
        
        # Check if we need to enforce limits
        limits = {
            MemoryType.WORKING: self.max_working_memory,
            MemoryType.EPISODIC: self.max_episodic_memory
        }
        
        if memory_type not in limits:
            return
        
        limit = limits[memory_type]
        if len(memories) <= limit:
            return
        
        # Sort by importance and remove least important
        memory_list = list(memories.values())
        memory_list.sort(key=lambda m: m.calculate_importance())
        
        memories_to_remove = len(memory_list) - limit
        for i in range(memories_to_remove):
            memory = memory_list[i]
            del memories[memory.memory_id]
            self.logger.debug(f"Removed low-importance {memory_type.value} memory: {memory.memory_id}")
    
    async def _update_memory_clusters(self, memory_id: str, memory: MemoryEntry) -> None:
        """Update memory clusters with new memory"""
        # Simple clustering based on tags and content similarity
        # In production, would use more sophisticated clustering algorithms
        
        # Find existing clusters that might be relevant
        relevant_clusters = []
        for cluster in self.clusters.values():
            # Check for tag overlap
            if any(tag in memory.tags for tag in cluster.theme.split()):
                relevant_clusters.append(cluster)
        
        if relevant_clusters:
            # Add to most relevant cluster
            best_cluster = max(relevant_clusters, key=lambda c: len(c.memories))
            best_cluster.add_memory(memory_id)
        else:
            # Create new cluster if memory has meaningful tags
            if memory.tags:
                cluster_id = f"cluster_{len(self.clusters)}"
                theme = "_".join(memory.tags[:3])  # Use first 3 tags as theme
                
                self.clusters[cluster_id] = MemoryCluster(
                    cluster_id=cluster_id,
                    theme=theme,
                    memories=[memory_id]
                )
    
    async def _extract_memory_patterns(self, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Extract patterns from a collection of memories"""
        patterns = {}
        
        # Frequency patterns
        tag_frequency = defaultdict(int)
        source_frequency = defaultdict(int)
        
        for memory in memories:
            for tag in memory.tags:
                tag_frequency[tag] += 1
            source_frequency[memory.source] += 1
        
        patterns["frequent_tags"] = dict(tag_frequency)
        patterns["source_distribution"] = dict(source_frequency)
        
        # Confidence patterns
        confidences = [m.confidence for m in memories]
        patterns["confidence_stats"] = {
            "mean": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences)
        }
        
        # Temporal patterns
        timestamps = [m.created_at for m in memories]
        if len(timestamps) > 1:
            time_span = max(timestamps) - min(timestamps)
            patterns["temporal_span_hours"] = time_span / 3600
        
        return patterns
    
    async def _analyze_semantic_patterns(self, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Analyze semantic memory for knowledge patterns"""
        return {
            "knowledge_domains": self._extract_knowledge_domains(memories),
            "concept_relationships": self._extract_concept_relationships(memories),
            "confidence_distribution": self._analyze_confidence_distribution(memories)
        }
    
    async def _analyze_procedural_patterns(self, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Analyze procedural memory for skill patterns"""
        return {
            "skill_categories": self._extract_skill_categories(memories),
            "performance_trends": self._analyze_performance_trends(memories),
            "success_patterns": self._extract_success_patterns(memories)
        }
    
    async def _analyze_meta_patterns(self, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Analyze meta memory for reasoning patterns"""
        return {
            "reasoning_strategies": self._extract_reasoning_strategies(memories),
            "decision_patterns": self._analyze_decision_patterns(memories),
            "learning_insights": self._extract_learning_insights(memories)
        }
    
    def _extract_knowledge_domains(self, memories: List[MemoryEntry]) -> Dict[str, int]:
        """Extract knowledge domains from semantic memories"""
        domains = defaultdict(int)
        for memory in memories:
            for tag in memory.tags:
                domains[tag] += 1
        return dict(domains)
    
    def _extract_concept_relationships(self, memories: List[MemoryEntry]) -> List[Dict[str, Any]]:
        """Extract concept relationships from memories"""
        relationships = []
        # Simplified relationship extraction
        for memory in memories:
            if "relationships" in memory.content:
                relationships.extend(memory.content["relationships"])
        return relationships
    
    def _analyze_confidence_distribution(self, memories: List[MemoryEntry]) -> Dict[str, float]:
        """Analyze confidence distribution"""
        confidences = [m.confidence for m in memories]
        return {
            "mean": sum(confidences) / len(confidences) if confidences else 0.0,
            "std": 0.0,  # Simplified - would calculate actual std dev
            "min": min(confidences) if confidences else 0.0,
            "max": max(confidences) if confidences else 0.0
        }
    
    def _extract_skill_categories(self, memories: List[MemoryEntry]) -> Dict[str, int]:
        """Extract skill categories from procedural memories"""
        categories = defaultdict(int)
        for memory in memories:
            skill_type = memory.content.get("skill_type", "general")
            categories[skill_type] += 1
        return dict(categories)
    
    def _analyze_performance_trends(self, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Analyze performance trends in procedural memories"""
        return {"trend": "stable"}  # Simplified implementation
    
    def _extract_success_patterns(self, memories: List[MemoryEntry]) -> List[Dict[str, Any]]:
        """Extract success patterns from procedural memories"""
        patterns = []
        for memory in memories:
            if memory.content.get("success", False):
                patterns.append({
                    "pattern": memory.content.get("pattern_type", "unknown"),
                    "confidence": memory.confidence
                })
        return patterns
    
    def _extract_reasoning_strategies(self, memories: List[MemoryEntry]) -> Dict[str, int]:
        """Extract reasoning strategies from meta memories"""
        strategies = defaultdict(int)
        for memory in memories:
            strategy = memory.content.get("reasoning_strategy", "default")
            strategies[strategy] += 1
        return dict(strategies)
    
    def _analyze_decision_patterns(self, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Analyze decision patterns in meta memories"""
        return {"pattern_count": len(memories)}  # Simplified
    
    def _extract_learning_insights(self, memories: List[MemoryEntry]) -> List[Dict[str, Any]]:
        """Extract learning insights from meta memories"""
        insights = []
        for memory in memories:
            if "insight" in memory.content:
                insights.append(memory.content["insight"])
        return insights
    
    async def _calculate_content_similarity(self, query: str, content: Dict[str, Any]) -> float:
        """Calculate similarity between query and memory content"""
        # Simplified similarity calculation
        # In production, would use semantic embeddings
        query_words = set(query.lower().split())
        content_text = json.dumps(content).lower()
        content_words = set(content_text.split())
        
        if not query_words or not content_words:
            return 0.0
        
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def _cleanup_empty_clusters(self) -> None:
        """Remove clusters with no memories"""
        empty_clusters = [
            cluster_id for cluster_id, cluster in self.clusters.items()
            if not cluster.memories
        ]
        
        for cluster_id in empty_clusters:
            del self.clusters[cluster_id]
    
    async def _periodic_consolidation(self) -> None:
        """Periodic memory consolidation task"""
        while True:
            try:
                await asyncio.sleep(self.consolidation_interval)
                
                # Run cleanup
                await self.cleanup_expired_memories()
                
                # Extract patterns
                await self.extract_learned_patterns()
                
                self.metrics["consolidation_runs"] += 1
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Consolidation task error: {e}")


__all__ = [
    'IntelligentMemoryManager',
    'MemoryType',
    'MemoryPriority', 
    'MemoryEntry',
    'MemoryCluster'
]
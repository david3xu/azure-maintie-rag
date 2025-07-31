"""
Context Manager - Manages conversation, session, and domain context for intelligent agents.
Provides temporal context tracking and efficient context retrieval for agent reasoning.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import json

from .agent_interface import AgentContext

logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context maintained by the context manager"""
    CONVERSATION = "conversation"
    SESSION = "session"
    DOMAIN = "domain"
    TEMPORAL = "temporal"
    PERFORMANCE = "performance"


class ContextScope(Enum):
    """Scope levels for context data"""
    GLOBAL = "global"
    USER = "user"
    SESSION = "session"
    CONVERSATION = "conversation"


@dataclass
class ContextEntry:
    """Individual context entry with metadata"""
    context_id: str
    context_type: ContextType
    scope: ContextScope
    data: Dict[str, Any]
    timestamp: float
    expiry_time: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if context entry has expired"""
        if self.expiry_time is None:
            return False
        return time.time() > self.expiry_time
    
    def access(self) -> None:
        """Record context access for tracking"""
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class ConversationTurn:
    """Single conversation turn with agent reasoning context"""
    turn_id: str
    user_query: str
    agent_response: Dict[str, Any]
    reasoning_trace: List[Dict[str, Any]]
    timestamp: float
    domain: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    sources: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SessionContext:
    """Session-level context with conversation history"""
    session_id: str
    user_id: Optional[str]
    start_time: float
    last_activity: float
    conversation_turns: deque = field(default_factory=lambda: deque(maxlen=50))
    preferences: Dict[str, Any] = field(default_factory=dict)
    domain_history: List[str] = field(default_factory=list)
    performance_baseline: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_turn(self, turn: ConversationTurn) -> None:
        """Add conversation turn and update session state"""
        self.conversation_turns.append(turn)
        self.last_activity = time.time()
        
        # Track domain usage
        if turn.domain and turn.domain not in self.domain_history:
            self.domain_history.append(turn.domain)


class ContextManager:
    """
    Manages multi-level context for intelligent agents with temporal tracking.
    
    Features:
    - Hierarchical context storage (global, user, session, conversation)
    - Temporal context tracking with automatic expiry
    - Efficient context retrieval and aggregation
    - Performance-aware context optimization
    - Memory-efficient LRU-based cleanup
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize context manager with data-driven configuration.
        
        Args:
            config: Context manager configuration
                - max_contexts_per_type: Maximum contexts per type
                - default_expiry_hours: Default context expiry time
                - cleanup_interval_seconds: How often to run cleanup
                - max_conversation_turns: Maximum turns to store per session
                - performance_tracking: Enable performance context tracking
        """
        self.config = config
        self.max_contexts_per_type = config.get("max_contexts_per_type", 1000)
        self.default_expiry_hours = config.get("default_expiry_hours", 24)
        self.cleanup_interval = config.get("cleanup_interval_seconds", 300)  # 5 minutes
        self.max_conversation_turns = config.get("max_conversation_turns", 50)
        self.performance_tracking = config.get("performance_tracking", True)
        
        # Context storage organized by type and scope
        self.contexts: Dict[ContextType, Dict[str, ContextEntry]] = {
            context_type: {} for context_type in ContextType
        }
        
        # Session management
        self.active_sessions: Dict[str, SessionContext] = {}
        
        # Domain context cache for fast retrieval
        self.domain_cache: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.performance_metrics = {
            "context_retrievals": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cleanup_runs": 0,
            "contexts_expired": 0
        }
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
    async def initialize(self) -> None:
        """Initialize context manager and start background tasks"""
        self.logger.info("Initializing context manager")
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        self.logger.info(f"Context manager initialized with {len(ContextType)} context types")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown context manager"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Context manager shutdown complete")
    
    async def create_agent_context(
        self, 
        query: str, 
        session_id: str,
        domain: Optional[str] = None,
        user_id: Optional[str] = None,
        search_constraints: Optional[Dict[str, Any]] = None,
        performance_targets: Optional[Dict[str, Any]] = None
    ) -> AgentContext:
        """
        Create comprehensive agent context from all available sources.
        
        Args:
            query: User query
            session_id: Session identifier
            domain: Optional domain context
            user_id: Optional user identifier  
            search_constraints: Optional search constraints
            performance_targets: Optional performance targets
            
        Returns:
            AgentContext with aggregated context data
        """
        async with self._lock:
            self.performance_metrics["context_retrievals"] += 1
            
            # Get or create session context
            session_context = await self._get_or_create_session(session_id, user_id)
            
            # Retrieve conversation history
            conversation_history = self._extract_conversation_history(session_context)
            
            # Get domain context if specified
            domain_context = None
            if domain:
                domain_context = await self._get_domain_context(domain)
            
            # Aggregate performance context
            performance_context = await self._get_performance_context(session_id, domain)
            
            # Build agent context
            agent_context = AgentContext(
                query=query,
                domain=domain,
                conversation_history=conversation_history,
                search_constraints=search_constraints or {},
                performance_targets=performance_targets or self._get_default_performance_targets(),
                metadata={
                    "session_id": session_id,
                    "user_id": user_id,
                    "domain_context": domain_context,
                    "performance_context": performance_context,
                    "session_turn_count": len(session_context.conversation_turns),
                    "timestamp": time.time()
                }
            )
            
            return agent_context
    
    async def save_conversation_turn(
        self,
        session_id: str,
        query: str,  
        response: Dict[str, Any],
        reasoning_trace: List[Dict[str, Any]],
        domain: Optional[str] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
        confidence: float = 0.0,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Save conversation turn to session context and update related contexts.
        
        Args:
            session_id: Session identifier
            query: Original user query
            response: Agent response
            reasoning_trace: Reasoning process trace
            domain: Domain context
            performance_metrics: Performance metrics
            confidence: Response confidence score
            sources: Information sources used
            
        Returns:
            Turn ID for the saved conversation turn
        """
        async with self._lock:
            session_context = self.active_sessions.get(session_id)
            if not session_context:
                self.logger.warning(f"No session context found for {session_id}")
                return ""
            
            # Create conversation turn
            turn_id = f"{session_id}_{len(session_context.conversation_turns)}"
            turn = ConversationTurn(
                turn_id=turn_id,
                user_query=query,
                agent_response=response,
                reasoning_trace=reasoning_trace,
                timestamp=time.time(),
                domain=domain,
                performance_metrics=performance_metrics or {},
                confidence=confidence,
                sources=sources or []
            )
            
            # Add to session
            session_context.add_turn(turn)
            
            # Update domain context if applicable
            if domain and self.performance_tracking:
                await self._update_domain_performance(domain, performance_metrics or {}, confidence)
            
            # Update temporal context patterns
            await self._update_temporal_patterns(session_id, turn)
            
            self.logger.debug(f"Saved conversation turn {turn_id} for session {session_id}")
            return turn_id
    
    async def get_domain_context(self, domain: str) -> Dict[str, Any]:
        """
        Get cached domain context with performance data.
        
        Args:
            domain: Domain identifier
            
        Returns:
            Domain context dictionary
        """
        if domain in self.domain_cache:
            self.performance_metrics["cache_hits"] += 1
            return self.domain_cache[domain]
        
        self.performance_metrics["cache_misses"] += 1
        context = await self._load_domain_context(domain)
        
        # Cache for future use
        self.domain_cache[domain] = context
        return context
    
    async def update_session_preferences(
        self, 
        session_id: str, 
        preferences: Dict[str, Any]
    ) -> bool:
        """
        Update session-level user preferences.
        
        Args:
            session_id: Session identifier
            preferences: Preference updates
            
        Returns:
            True if update successful
        """
        async with self._lock:
            session_context = self.active_sessions.get(session_id)
            if not session_context:
                return False
            
            session_context.preferences.update(preferences)
            return True
    
    async def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """
        Get analytics for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session analytics dictionary
        """
        session_context = self.active_sessions.get(session_id)
        if not session_context:
            return {}
        
        turns = list(session_context.conversation_turns)
        if not turns:
            return {"turn_count": 0}
        
        # Calculate analytics
        avg_confidence = sum(turn.confidence for turn in turns) / len(turns)
        avg_response_time = 0
        if turns and all(turn.performance_metrics.get("total_time_ms") for turn in turns):
            avg_response_time = sum(
                turn.performance_metrics["total_time_ms"] for turn in turns
            ) / len(turns)
        
        domain_distribution = defaultdict(int)
        for turn in turns:
            if turn.domain:
                domain_distribution[turn.domain] += 1
        
        return {
            "session_id": session_id,
            "turn_count": len(turns),
            "session_duration_seconds": time.time() - session_context.start_time,
            "avg_confidence": avg_confidence,
            "avg_response_time_ms": avg_response_time,
            "domain_distribution": dict(domain_distribution),
            "unique_domains": len(domain_distribution),
            "last_activity": session_context.last_activity
        }
    
    async def cleanup_expired_contexts(self) -> Dict[str, int]:
        """
        Clean up expired contexts and return cleanup statistics.
        
        Returns:
            Dictionary with cleanup statistics
        """
        async with self._lock:
            cleanup_stats = {
                "contexts_removed": 0,
                "sessions_cleaned": 0,
                "cache_cleared": 0
            }
            
            # Clean up expired context entries
            for context_type, contexts in self.contexts.items():
                expired_keys = [
                    key for key, entry in contexts.items() 
                    if entry.is_expired()
                ]
                
                for key in expired_keys:
                    del contexts[key]
                    cleanup_stats["contexts_removed"] += 1
            
            # Clean up old sessions (no activity for 24 hours)
            cutoff_time = time.time() - (24 * 3600)
            expired_sessions = [
                session_id for session_id, session in self.active_sessions.items()
                if session.last_activity < cutoff_time
            ]
            
            for session_id in expired_sessions:
                del self.active_sessions[session_id]
                cleanup_stats["sessions_cleaned"] += 1
            
            # Clear domain cache if too large
            if len(self.domain_cache) > 100:
                self.domain_cache.clear()
                cleanup_stats["cache_cleared"] = 1
            
            self.performance_metrics["cleanup_runs"] += 1
            self.performance_metrics["contexts_expired"] += cleanup_stats["contexts_removed"]
            
            if cleanup_stats["contexts_removed"] > 0:
                self.logger.info(f"Cleanup completed: {cleanup_stats}")
            
            return cleanup_stats
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get context manager performance metrics"""
        return {
            **self.performance_metrics,
            "active_sessions": len(self.active_sessions),
            "total_contexts": sum(len(contexts) for contexts in self.contexts.values()),
            "domain_cache_size": len(self.domain_cache),
            "cache_hit_rate": (
                self.performance_metrics["cache_hits"] / 
                max(1, self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"])
            )
        }
    
    # Private implementation methods
    
    async def _get_or_create_session(self, session_id: str, user_id: Optional[str]) -> SessionContext:
        """Get existing session or create new one"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Create new session
        session = SessionContext(
            session_id=session_id,
            user_id=user_id,
            start_time=time.time(),
            last_activity=time.time()
        )
        
        self.active_sessions[session_id] = session
        self.logger.debug(f"Created new session context: {session_id}")
        return session
    
    def _extract_conversation_history(self, session: SessionContext) -> List[Dict[str, Any]]:
        """Extract conversation history for agent context"""
        history = []
        for turn in session.conversation_turns:
            history.append({
                "query": turn.user_query,
                "response": turn.agent_response,
                "timestamp": turn.timestamp,
                "domain": turn.domain,
                "confidence": turn.confidence
            })
        return history
    
    async def _get_domain_context(self, domain: str) -> Dict[str, Any]:
        """Get domain-specific context"""
        return await self.get_domain_context(domain)
    
    async def _get_performance_context(self, session_id: str, domain: Optional[str]) -> Dict[str, Any]:
        """Get performance context for optimization"""
        context = {
            "session_performance": {},
            "domain_performance": {},
            "global_baseline": {}
        }
        
        # Session performance
        session = self.active_sessions.get(session_id)
        if session:
            context["session_performance"] = session.performance_baseline
        
        # Domain performance
        if domain and domain in self.domain_cache:
            context["domain_performance"] = self.domain_cache[domain].get("performance", {})
        
        return context
    
    def _get_default_performance_targets(self) -> Dict[str, Any]:
        """Get default performance targets"""
        return {
            "max_response_time": 3.0,
            "min_confidence": 0.7,
            "max_memory_mb": 200
        }
    
    async def _load_domain_context(self, domain: str) -> Dict[str, Any]:
        """Load domain context from storage"""
        # Placeholder - would load from persistent storage
        return {
            "domain": domain,
            "patterns": {},
            "performance": {
                "avg_response_time": 2.0,
                "avg_confidence": 0.8
            },
            "loaded_at": time.time()
        }
    
    async def _update_domain_performance(
        self, 
        domain: str, 
        metrics: Dict[str, Any], 
        confidence: float
    ) -> None:
        """Update domain performance tracking"""
        if domain not in self.domain_cache:
            await self.get_domain_context(domain)
        
        perf = self.domain_cache[domain].setdefault("performance", {})
        perf["last_confidence"] = confidence
        perf["last_updated"] = time.time()
        
        if "total_time_ms" in metrics:
            perf["last_response_time"] = metrics["total_time_ms"]
    
    async def _update_temporal_patterns(self, session_id: str, turn: ConversationTurn) -> None:
        """Update temporal context patterns (Graphiti insight)"""
        # Implement temporal pattern tracking
        temporal_context = {
            "session_id": session_id,
            "turn_timestamp": turn.timestamp,
            "domain": turn.domain,
            "confidence": turn.confidence,
            "query_complexity": len(turn.user_query.split())
        }
        
        # Store temporal context entry
        context_id = f"temporal_{session_id}_{turn.turn_id}"
        entry = ContextEntry(
            context_id=context_id,
            context_type=ContextType.TEMPORAL,
            scope=ContextScope.SESSION,
            data=temporal_context,
            timestamp=turn.timestamp,
            expiry_time=time.time() + (7 * 24 * 3600)  # 7 days
        )
        
        self.contexts[ContextType.TEMPORAL][context_id] = entry
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup task"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_expired_contexts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")


__all__ = [
    'ContextManager',
    'ContextType', 
    'ContextScope',
    'ContextEntry',
    'ConversationTurn',
    'SessionContext'
]
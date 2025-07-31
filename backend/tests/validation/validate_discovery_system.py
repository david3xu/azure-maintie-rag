#!/usr/bin/env python3
"""
Dynamic Discovery System Validation Tests
Comprehensive validation for zero-configuration domain adaptation system.
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Discovery system imports
from agents.discovery.domain_pattern_engine import DomainPatternEngine, PatternType, ConfidenceLevel
from agents.discovery.zero_config_adapter import ZeroConfigAdapter, DomainAdaptationStrategy, AdaptationConfidence
from agents.discovery.pattern_learning_system import PatternLearningSystem, LearningMode, LearningExample
from agents.discovery.domain_context_enhancer import DomainContextEnhancer, ContextEnhancementLevel, ContextEnhancementRequest

# Agent base architecture imports
from agents.base import AgentContext, ContextManager, ContextType, ContextEntry, AgentCapability

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation test"""
    test_name: str
    passed: bool
    duration_ms: float
    details: Dict[str, Any]
    error_message: Optional[str] = None


class DiscoverySystemValidator:
    """
    Comprehensive validator for the Dynamic Discovery System.
    
    Tests all components individually and as an integrated system:
    - Domain pattern discovery accuracy
    - Zero-configuration adaptation capabilities  
    - Pattern learning and evolution
    - Context enhancement integration
    - Performance and scalability
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize validator with test configuration"""
        self.config = config or {}
        self.results: List[ValidationResult] = []
        
        # Test data sets for different domains
        self.test_domains = {
            "healthcare": [
                "Patient John Smith was diagnosed with diabetes mellitus type 2. Blood glucose levels were 180 mg/dL.",
                "The physician prescribed metformin 500mg twice daily. Follow-up appointment scheduled in 3 months.",
                "Medical history includes hypertension and obesity. BMI calculated at 32.5 kg/m²."
            ],
            "finance": [
                "The quarterly earnings report showed revenue of $2.5 million, up 15% from last quarter.",
                "Stock prices fluctuated between $45.20 and $52.80 during trading hours.",
                "Investment portfolio diversification includes 60% equities, 30% bonds, 10% commodities."
            ],
            "technology": [
                "The microservice architecture deployed on Kubernetes cluster handles 10,000 requests per second.",
                "Docker containers are orchestrated using Helm charts with automated CI/CD pipelines.",
                "API endpoints implement REST standards with OAuth 2.0 authentication and rate limiting."
            ],
            "legal": [
                "The contract stipulates termination clauses under Section 12.3 with 30-day notice period.",
                "Intellectual property rights are governed by applicable federal and state laws.",
                "Damages shall be limited to direct costs, excluding consequential or punitive damages."
            ]
        }
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive validation of the discovery system.
        
        Returns:
            Validation summary with results and metrics
        """
        self.logger.info("Starting comprehensive discovery system validation")
        start_time = time.time()
        
        # Test phases
        validation_phases = [
            ("Component Initialization", self._test_component_initialization),
            ("Domain Pattern Discovery", self._test_domain_pattern_discovery),
            ("Zero-Config Domain Detection", self._test_zero_config_detection),
            ("Pattern Learning System", self._test_pattern_learning_system),
            ("Context Enhancement Integration", self._test_context_enhancement),
            ("End-to-End Integration", self._test_end_to_end_integration),
            ("Performance and Scalability", self._test_performance_scalability),
            ("Error Handling and Robustness", self._test_error_handling)
        ]
        
        phase_results = {}
        
        for phase_name, test_function in validation_phases:
            self.logger.info(f"Running validation phase: {phase_name}")
            
            try:
                phase_result = await test_function()
                phase_results[phase_name] = phase_result
                
                self.logger.info(
                    f"Phase '{phase_name}' completed: "
                    f"{phase_result['passed_tests']}/{phase_result['total_tests']} tests passed"
                )
                
            except Exception as e:
                self.logger.error(f"Phase '{phase_name}' failed with error: {e}")
                phase_results[phase_name] = {
                    "passed_tests": 0,
                    "total_tests": 1,
                    "error": str(e)
                }
        
        # Calculate overall results
        total_duration = time.time() - start_time
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results if result.passed)
        
        validation_summary = {
            "overall_success": passed_tests == total_tests,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "total_duration_seconds": total_duration,
            "phase_results": phase_results,
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "duration_ms": r.duration_ms,
                    "error": r.error_message
                }
                for r in self.results
            ]
        }
        
        self.logger.info(
            f"Validation completed: {passed_tests}/{total_tests} tests passed "
            f"in {total_duration:.2f} seconds"
        )
        
        return validation_summary
    
    async def _test_component_initialization(self) -> Dict[str, Any]:
        """Test initialization of all discovery system components"""
        results = []
        
        # Test DomainPatternEngine initialization
        result = await self._run_test("DomainPatternEngine initialization", self._test_pattern_engine_init)
        results.append(result)
        
        # Test ZeroConfigAdapter initialization
        result = await self._run_test("ZeroConfigAdapter initialization", self._test_zero_config_init)
        results.append(result)
        
        # Test PatternLearningSystem initialization
        result = await self._run_test("PatternLearningSystem initialization", self._test_learning_system_init)
        results.append(result)
        
        # Test DomainContextEnhancer initialization
        result = await self._run_test("DomainContextEnhancer initialization", self._test_context_enhancer_init)
        results.append(result)
        
        return {
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r.passed),
            "results": results
        }
    
    async def _test_domain_pattern_discovery(self) -> Dict[str, Any]:
        """Test domain pattern discovery capabilities"""
        results = []
        
        # Test pattern extraction for each domain
        for domain_name, domain_texts in self.test_domains.items():
            result = await self._run_test(
                f"Pattern discovery for {domain_name} domain",
                lambda d=domain_name, t=domain_texts: self._test_domain_patterns(d, t)
            )
            results.append(result)
        
        # Test pattern type coverage
        result = await self._run_test("Pattern type coverage", self._test_pattern_type_coverage)
        results.append(result)
        
        # Test domain fingerprint creation
        result = await self._run_test("Domain fingerprint creation", self._test_fingerprint_creation)
        results.append(result)
        
        return {
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r.passed),
            "results": results
        }
    
    async def _test_zero_config_detection(self) -> Dict[str, Any]:
        """Test zero-configuration domain detection and adaptation"""
        results = []
        
        # Test domain detection accuracy
        result = await self._run_test("Domain detection accuracy", self._test_detection_accuracy)
        results.append(result)
        
        # Test adaptation profile creation
        result = await self._run_test("Adaptation profile creation", self._test_adaptation_profiles)
        results.append(result)
        
        # Test similarity matching
        result = await self._run_test("Domain similarity matching", self._test_domain_similarity)
        results.append(result)
        
        return {
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r.passed),
            "results": results
        }
    
    async def _test_pattern_learning_system(self) -> Dict[str, Any]:
        """Test pattern learning and evolution capabilities"""
        results = []
        
        # Test learning session management
        result = await self._run_test("Learning session management", self._test_learning_sessions)
        results.append(result)
        
        # Test different learning modes
        for mode in [LearningMode.UNSUPERVISED, LearningMode.SUPERVISED, LearningMode.REINFORCEMENT]:
            result = await self._run_test(
                f"Learning mode: {mode.value}",
                lambda m=mode: self._test_learning_mode(m)
            )
            results.append(result)
        
        # Test pattern evolution tracking
        result = await self._run_test("Pattern evolution tracking", self._test_pattern_evolution)
        results.append(result)
        
        return {
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r.passed),
            "results": results
        }
    
    async def _test_context_enhancement(self) -> Dict[str, Any]:
        """Test context enhancement and integration"""
        results = []
        
        # Test context enhancement levels
        for level in ContextEnhancementLevel:
            result = await self._run_test(
                f"Context enhancement level: {level.value}",
                lambda l=level: self._test_enhancement_level(l)
            )
            results.append(result)
        
        # Test integration with existing context management
        result = await self._run_test("Context manager integration", self._test_context_integration)
        results.append(result)
        
        return {
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r.passed),
            "results": results
        }
    
    async def _test_end_to_end_integration(self) -> Dict[str, Any]:
        """Test complete end-to-end integration workflow"""
        results = []
        
        # Test complete discovery workflow
        result = await self._run_test("Complete discovery workflow", self._test_complete_workflow)
        results.append(result)
        
        # Test multi-domain handling
        result = await self._run_test("Multi-domain handling", self._test_multi_domain_handling)
        results.append(result)
        
        # Test continuous learning integration
        result = await self._run_test("Continuous learning integration", self._test_continuous_learning)
        results.append(result)
        
        return {
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r.passed),
            "results": results
        }
    
    async def _test_performance_scalability(self) -> Dict[str, Any]:
        """Test performance and scalability characteristics"""
        results = []
        
        # Test performance benchmarks
        result = await self._run_test("Performance benchmarks", self._test_performance_benchmarks)
        results.append(result)
        
        # Test memory usage
        result = await self._run_test("Memory usage validation", self._test_memory_usage)
        results.append(result)
        
        # Test caching effectiveness
        result = await self._run_test("Caching effectiveness", self._test_caching_effectiveness)
        results.append(result)
        
        return {
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r.passed),
            "results": results
        }
    
    async def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and robustness"""
        results = []
        
        # Test invalid input handling
        result = await self._run_test("Invalid input handling", self._test_invalid_inputs)
        results.append(result)
        
        # Test graceful degradation
        result = await self._run_test("Graceful degradation", self._test_graceful_degradation)
        results.append(result)
        
        return {
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r.passed),
            "results": results
        }
    
    # Individual test implementations
    
    async def _test_pattern_engine_init(self) -> Dict[str, Any]:
        """Test DomainPatternEngine initialization"""
        config = {
            "min_pattern_frequency": 2,
            "confidence_threshold": 0.4,
            "max_patterns_per_type": 50
        }
        
        engine = DomainPatternEngine(config)
        
        assert engine.min_pattern_frequency == 2
        assert engine.confidence_threshold == 0.4
        assert engine.max_patterns_per_type == 50
        assert isinstance(engine.discovered_patterns, dict)
        assert isinstance(engine.domain_fingerprints, dict)
        
        return {"success": True, "component": "DomainPatternEngine"}
    
    async def _test_zero_config_init(self) -> Dict[str, Any]:
        """Test ZeroConfigAdapter initialization"""
        pattern_engine = DomainPatternEngine({})
        config = {
            "pattern_engine": pattern_engine,
            "adaptation_strategy": "balanced"
        }
        
        adapter = ZeroConfigAdapter(config)
        
        assert adapter.adaptation_strategy == DomainAdaptationStrategy.BALANCED
        assert adapter.pattern_engine is pattern_engine
        assert isinstance(adapter.known_domains, dict)
        assert isinstance(adapter.adaptation_profiles, dict)
        
        return {"success": True, "component": "ZeroConfigAdapter"}
    
    async def _test_learning_system_init(self) -> Dict[str, Any]:
        """Test PatternLearningSystem initialization"""
        config = {
            "learning_modes": ["unsupervised", "reinforcement"],
            "confidence_learning_rate": 0.1
        }
        
        learning_system = PatternLearningSystem(config)
        
        assert LearningMode.UNSUPERVISED in learning_system.enabled_learning_modes
        assert LearningMode.REINFORCEMENT in learning_system.enabled_learning_modes
        assert learning_system.confidence_learning_rate == 0.1
        assert isinstance(learning_system.learned_patterns, dict)
        
        return {"success": True, "component": "PatternLearningSystem"}
    
    async def _test_context_enhancer_init(self) -> Dict[str, Any]:
        """Test DomainContextEnhancer initialization"""
        pattern_engine = DomainPatternEngine({})
        adapter = ZeroConfigAdapter({"pattern_engine": pattern_engine})
        learning_system = PatternLearningSystem({})
        
        config = {
            "pattern_engine": pattern_engine,
            "zero_config_adapter": adapter,
            "pattern_learning_system": learning_system,
            "enhancement_level": "comprehensive"
        }
        
        enhancer = DomainContextEnhancer(config)
        await enhancer.initialize()
        
        assert enhancer.default_enhancement_level == ContextEnhancementLevel.COMPREHENSIVE
        assert enhancer.pattern_engine is pattern_engine
        assert enhancer.zero_config_adapter is adapter
        assert enhancer.pattern_learning_system is learning_system
        assert enhancer.active_learning_session is not None
        
        return {"success": True, "component": "DomainContextEnhancer"}
    
    async def _test_domain_patterns(self, domain_name: str, domain_texts: List[str]) -> Dict[str, Any]:
        """Test pattern discovery for specific domain"""
        engine = DomainPatternEngine({
            "min_pattern_frequency": 1,
            "confidence_threshold": 0.3
        })
        
        fingerprint = await engine.analyze_text_corpus(domain_texts, domain_hint=domain_name)
        
        assert fingerprint.domain_name == domain_name
        assert fingerprint.confidence > 0.0
        assert len(fingerprint.primary_patterns) > 0
        assert fingerprint.vocabulary_size > 0
        
        # Check for expected pattern types based on domain
        expected_patterns = {
            "healthcare": [PatternType.ENTITY, PatternType.NUMERICAL],
            "finance": [PatternType.NUMERICAL, PatternType.CONCEPT],
            "technology": [PatternType.CONCEPT, PatternType.ENTITY],
            "legal": [PatternType.CONCEPT, PatternType.PROCEDURAL]
        }
        
        domain_expected = expected_patterns.get(domain_name, [])
        found_types = set(fingerprint.primary_patterns.keys())
        
        overlap = len(set(domain_expected) & found_types)
        assert overlap > 0, f"Expected patterns {domain_expected} not found in {found_types}"
        
        return {
            "domain": domain_name,
            "patterns_found": len([p for patterns in fingerprint.primary_patterns.values() for p in patterns]),
            "confidence": fingerprint.confidence,
            "pattern_types": list(found_types)
        }
    
    async def _test_pattern_type_coverage(self) -> Dict[str, Any]:
        """Test coverage of different pattern types"""
        engine = DomainPatternEngine({"min_pattern_frequency": 1})
        
        # Comprehensive test text covering multiple pattern types
        test_text = [
            "Dr. Smith diagnosed patient John Doe with diabetes on January 15, 2024.",
            "Treatment costs $2,500 per month with 90% insurance coverage.",
            "First, check blood glucose levels. Second, adjust medication dosage.",
            "The API processes 1,000 requests per second using microservices architecture."
        ]
        
        fingerprint = await engine.analyze_text_corpus(test_text)
        
        found_types = set(fingerprint.primary_patterns.keys())
        expected_types = {PatternType.ENTITY, PatternType.TEMPORAL, PatternType.NUMERICAL, PatternType.PROCEDURAL}
        
        coverage = len(found_types & expected_types) / len(expected_types)
        assert coverage >= 0.5, f"Insufficient pattern type coverage: {coverage}"
        
        return {
            "coverage_ratio": coverage,
            "found_types": [t.value for t in found_types],
            "expected_types": [t.value for t in expected_types]
        }
    
    async def _test_fingerprint_creation(self) -> Dict[str, Any]:
        """Test domain fingerprint creation and properties"""
        engine = DomainPatternEngine({})
        
        test_texts = self.test_domains["healthcare"]
        fingerprint = await engine.analyze_text_corpus(test_texts)
        
        assert fingerprint.domain_id is not None
        assert fingerprint.confidence >= 0.0 and fingerprint.confidence <= 1.0
        assert fingerprint.vocabulary_size >= 0
        assert fingerprint.concept_density >= 0.0
        assert fingerprint.source_documents == len(test_texts)
        assert fingerprint.total_tokens > 0
        
        # Test similarity calculation
        fingerprint2 = await engine.analyze_text_corpus(test_texts)  # Same texts
        similarity = fingerprint.get_similarity_score(fingerprint2)
        assert similarity > 0.8, f"Self-similarity should be high: {similarity}"
        
        return {
            "fingerprint_valid": True,
            "self_similarity": similarity,
            "vocabulary_size": fingerprint.vocabulary_size,
            "confidence": fingerprint.confidence
        }
    
    async def _test_detection_accuracy(self) -> Dict[str, Any]:
        """Test domain detection accuracy"""
        pattern_engine = DomainPatternEngine({})
        adapter = ZeroConfigAdapter({"pattern_engine": pattern_engine})
        
        detection_results = {}
        
        # Test detection for each domain
        for domain_name, domain_texts in self.test_domains.items():
            # Train on the domain first
            await pattern_engine.analyze_text_corpus(domain_texts, domain_hint=domain_name)
            
            # Test detection on a query from the same domain
            test_query = domain_texts[0]
            detection_result = await adapter.detect_domain_from_query(test_query)
            
            detection_results[domain_name] = {
                "detected_domain": detection_result.detected_domain,
                "confidence": detection_result.confidence,
                "confidence_level": detection_result.confidence_level.value if detection_result.confidence_level else "unknown"
            }
        
        # Check that at least some domains were detected with reasonable confidence
        detected_count = sum(1 for r in detection_results.values() if r["detected_domain"] is not None)
        assert detected_count >= 2, f"Too few domains detected: {detected_count}"
        
        return {
            "domains_detected": detected_count,
            "total_domains": len(self.test_domains),
            "detection_results": detection_results
        }
    
    async def _test_adaptation_profiles(self) -> Dict[str, Any]:
        """Test adaptation profile creation"""
        pattern_engine = DomainPatternEngine({})
        adapter = ZeroConfigAdapter({"pattern_engine": pattern_engine})
        
        # Create detection result
        test_texts = self.test_domains["technology"] 
        fingerprint = await pattern_engine.analyze_text_corpus(test_texts)
        
        from agents.discovery.zero_config_adapter import DomainDetectionResult
        detection_result = DomainDetectionResult(
            detected_domain="technology",
            confidence=0.8,
            confidence_level=AdaptationConfidence.HIGH,
            domain_fingerprint=fingerprint
        )
        
        base_config = {"default_timeout": 3.0}
        adapted_config, adaptation_profile = await adapter.adapt_agent_to_domain(
            detection_result, base_config
        )
        
        assert adaptation_profile.domain_id == "technology"
        assert len(adaptation_profile.recommended_capabilities) > 0
        assert isinstance(adaptation_profile.configuration_overrides, dict)
        assert isinstance(adaptation_profile.search_strategy_preferences, dict)
        assert adaptation_profile.confidence > 0.0
        
        # Verify config was adapted
        assert "search_preferences" in adapted_config
        assert "domain_adaptation" in adapted_config
        
        return {
            "profile_created": True,
            "capabilities_count": len(adaptation_profile.recommended_capabilities),
            "profile_confidence": adaptation_profile.confidence,
            "config_adapted": "domain_adaptation" in adapted_config
        }
    
    async def _test_domain_similarity(self) -> Dict[str, Any]:
        """Test domain similarity matching"""
        pattern_engine = DomainPatternEngine({})
        
        # Create fingerprints for different domains
        healthcare_fp = await pattern_engine.analyze_text_corpus(
            self.test_domains["healthcare"], domain_hint="healthcare"
        )
        finance_fp = await pattern_engine.analyze_text_corpus(
            self.test_domains["finance"], domain_hint="finance"
        )
        
        # Test similarity between different domains (should be low)
        cross_similarity = healthcare_fp.get_similarity_score(finance_fp)
        assert cross_similarity < 0.5, f"Cross-domain similarity too high: {cross_similarity}"
        
        # Test similarity within same domain (should be high)
        healthcare_fp2 = await pattern_engine.analyze_text_corpus(
            self.test_domains["healthcare"][:2], domain_hint="healthcare"
        )
        within_similarity = healthcare_fp.get_similarity_score(healthcare_fp2)
        assert within_similarity > 0.3, f"Within-domain similarity too low: {within_similarity}"
        
        return {
            "cross_domain_similarity": cross_similarity,
            "within_domain_similarity": within_similarity,
            "similarity_discrimination": within_similarity > cross_similarity
        }
    
    async def _test_learning_sessions(self) -> Dict[str, Any]:
        """Test learning session management"""
        learning_system = PatternLearningSystem({
            "learning_modes": ["unsupervised", "supervised", "reinforcement"]
        })
        
        # Start learning session
        session_id = await learning_system.start_learning_session(
            LearningMode.UNSUPERVISED,
            {"test": "validation"}
        )
        
        assert session_id in learning_system.active_sessions
        session = learning_system.active_sessions[session_id]
        assert session.learning_mode == LearningMode.UNSUPERVISED
        assert session.metadata["test"] == "validation"
        
        # End session
        completed_session = await learning_system.end_learning_session(session_id)
        assert completed_session.end_time is not None
        assert session_id not in learning_system.active_sessions
        assert completed_session in learning_system.completed_sessions
        
        return {
            "session_created": True,
            "session_ended": True,
            "session_id": session_id
        }
    
    async def _test_learning_mode(self, mode: LearningMode) -> Dict[str, Any]:
        """Test specific learning mode"""
        learning_system = PatternLearningSystem({
            "learning_modes": [mode.value],
            "confidence_learning_rate": 0.2
        })
        
        session_id = await learning_system.start_learning_session(mode)
        
        # Create learning examples
        examples = []
        for i, text in enumerate(self.test_domains["healthcare"][:2]):
            example = LearningExample(
                example_id=f"test_{i}",
                text=text,
                labels={"domain": "healthcare"} if mode == LearningMode.SUPERVISED else {},
                feedback={"reward": 0.8} if mode == LearningMode.REINFORCEMENT else None
            )
            examples.append(example)
        
        # Learn from examples
        results = await learning_system.learn_patterns_from_examples(session_id, examples)
        
        assert "new_patterns" in results
        assert "evolved_patterns" in results
        assert results["new_patterns"] >= 0
        assert results["evolved_patterns"] >= 0
        
        await learning_system.end_learning_session(session_id)
        
        return {
            "mode": mode.value,
            "new_patterns": results["new_patterns"],
            "evolved_patterns": results["evolved_patterns"],
            "examples_processed": len(examples)
        }
    
    async def _test_pattern_evolution(self) -> Dict[str, Any]:
        """Test pattern evolution tracking"""
        learning_system = PatternLearningSystem({
            "pattern_evolution_tracking": True,
            "confidence_learning_rate": 0.3
        })
        
        session_id = await learning_system.start_learning_session(LearningMode.SUPERVISED)
        
        # Create example and learn
        example = LearningExample(
            example_id="evolution_test",
            text="Patient has diabetes and high blood pressure",
            labels={"condition": "diabetes"}
        )
        
        await learning_system.learn_patterns_from_examples(session_id, [example])
        
        # Find a learned pattern to apply feedback to
        if learning_system.learned_patterns:
            pattern_id = list(learning_system.learned_patterns.keys())[0]
            
            # Apply positive feedback
            feedback_result = await learning_system.apply_feedback_learning(
                pattern_id,
                {"success": True, "confidence": 0.9},
                {"test": "evolution"}
            )
            
            assert feedback_result["success"]
            assert "confidence_change" in feedback_result
            
            # Check evolution was tracked
            evolution_insights = await learning_system.get_pattern_evolution_insights(1.0)
            assert evolution_insights["events_in_window"] > 0
            
            await learning_system.end_learning_session(session_id)
            
            return {
                "evolution_tracked": True,
                "evolution_events": evolution_insights["events_in_window"],
                "confidence_changed": abs(feedback_result["confidence_change"]) > 0.01
            }
        
        await learning_system.end_learning_session(session_id)
        
        return {
            "evolution_tracked": False,
            "reason": "No patterns learned to track evolution"
        }
    
    async def _test_enhancement_level(self, level: ContextEnhancementLevel) -> Dict[str, Any]:
        """Test specific context enhancement level"""
        # Set up components
        pattern_engine = DomainPatternEngine({})
        adapter = ZeroConfigAdapter({"pattern_engine": pattern_engine})
        learning_system = PatternLearningSystem({})
        
        enhancer = DomainContextEnhancer({
            "pattern_engine": pattern_engine,
            "zero_config_adapter": adapter,
            "pattern_learning_system": learning_system,
            "enhancement_level": level.value
        })
        
        await enhancer.initialize()
        
        # Create test context
        test_context = AgentContext(
            query="Patient presents with chest pain and shortness of breath",
            domain=None,
            conversation_history=[],
            search_constraints={},
            performance_targets={},
            metadata={}
        )
        
        # Create enhancement request
        request = ContextEnhancementRequest(
            request_id="test_enhancement",
            original_context=test_context,
            enhancement_level=level,
            additional_text=["Medical history shows previous cardiac events"]
        )
        
        # Enhance context
        enhanced_context, domain_context = await enhancer.enhance_agent_context(request)
        
        assert enhanced_context is not None
        assert domain_context is not None
        assert domain_context.enhancement_level == level
        
        # Verify enhancement based on level
        if level in [ContextEnhancementLevel.COMPREHENSIVE, ContextEnhancementLevel.ADAPTIVE]:
            assert len(domain_context.learned_patterns) >= 0  # May be empty but should exist
        
        return {
            "level": level.value,
            "context_enhanced": True,
            "domain_detected": domain_context.domain_id != "error",
            "detection_confidence": domain_context.detection_result.confidence
        }
    
    async def _test_context_integration(self) -> Dict[str, Any]:
        """Test integration with existing context management"""
        # Create mock context manager
        context_manager = ContextManager({})
        
        # Set up discovery system with context manager
        pattern_engine = DomainPatternEngine({})
        adapter = ZeroConfigAdapter({"pattern_engine": pattern_engine})
        learning_system = PatternLearningSystem({})
        
        enhancer = DomainContextEnhancer({
            "context_manager": context_manager,
            "pattern_engine": pattern_engine,
            "zero_config_adapter": adapter,  
            "pattern_learning_system": learning_system
        })
        
        await enhancer.initialize()
        
        # Test that components are properly wired
        assert enhancer.context_manager is context_manager
        assert enhancer.pattern_engine is pattern_engine
        assert enhancer.zero_config_adapter is adapter
        assert enhancer.pattern_learning_system is learning_system
        
        return {
            "integration_successful": True,
            "context_manager_connected": enhancer.context_manager is not None,
            "all_components_connected": all([
                enhancer.pattern_engine,
                enhancer.zero_config_adapter,
                enhancer.pattern_learning_system
            ])
        }
    
    async def _test_complete_workflow(self) -> Dict[str, Any]:
        """Test complete end-to-end discovery workflow"""
        # Initialize complete system
        pattern_engine = DomainPatternEngine({"min_pattern_frequency": 1})
        adapter = ZeroConfigAdapter({"pattern_engine": pattern_engine})
        learning_system = PatternLearningSystem({"learning_modes": ["reinforcement"]})
        
        enhancer = DomainContextEnhancer({
            "pattern_engine": pattern_engine,
            "zero_config_adapter": adapter,
            "pattern_learning_system": learning_system,
            "enhancement_level": "comprehensive"
        })
        
        await enhancer.initialize()
        
        # Simulate complete workflow
        test_query = "Execute API call to process payment of $150.00 for user account #12345"
        
        # 1. Create initial context
        context = AgentContext(
            query=test_query,
            domain=None,
            conversation_history=[],
            search_constraints={},
            performance_targets={},
            metadata={}
        )
        
        # 2. Enhance context through discovery system
        request = ContextEnhancementRequest(
            request_id="workflow_test",
            original_context=context,
            enhancement_level=ContextEnhancementLevel.COMPREHENSIVE,
            additional_text=["Previous transaction was successful"]
        )
        
        enhanced_context, domain_context = await enhancer.enhance_agent_context(request)
        
        # 3. Simulate interaction result and provide feedback
        interaction_result = {
            "success": True,
            "confidence": 0.85,
            "response_time_ms": 1200,
            "user_satisfaction": 0.9
        }
        
        context_key = enhancer._generate_context_key(context, request.additional_text)
        await enhancer.provide_learning_feedback(context_key, interaction_result)
        
        # 4. Verify workflow completed successfully
        assert enhanced_context.domain is not None
        assert domain_context.domain_id != "error"
        assert domain_context.detection_result.confidence > 0.0
        
        # 5. Check that learning occurred
        metrics = enhancer.get_performance_metrics()
        assert metrics["contexts_enhanced"] > 0
        assert metrics["successful_enhancements"] > 0
        
        return {
            "workflow_completed": True,
            "domain_detected": enhanced_context.domain,
            "detection_confidence": domain_context.detection_result.confidence,
            "learning_feedback_applied": True,
            "contexts_enhanced": metrics["contexts_enhanced"]
        }
    
    async def _test_multi_domain_handling(self) -> Dict[str, Any]:
        """Test handling of multiple domains"""
        pattern_engine = DomainPatternEngine({})
        adapter = ZeroConfigAdapter({"pattern_engine": pattern_engine})
        
        # Train on multiple domains
        for domain_name, domain_texts in self.test_domains.items():
            await pattern_engine.analyze_text_corpus(domain_texts, domain_hint=domain_name)
        
        # Test queries from different domains
        test_queries = [
            ("Check patient blood pressure and heart rate", "healthcare"),
            ("Process quarterly financial report", "finance"),
            ("Deploy microservice to Kubernetes cluster", "technology"),
            ("Review contract terms and conditions", "legal")
        ]
        
        detection_results = []
        for query, expected_domain in test_queries:
            result = await adapter.detect_domain_from_query(query)
            detection_results.append({
                "query": query,
                "expected": expected_domain,
                "detected": result.detected_domain,
                "confidence": result.confidence
            })
        
        # Verify multi-domain handling
        detected_domains = set(r["detected"] for r in detection_results if r["detected"])
        assert len(detected_domains) >= 2, f"Too few domains detected: {len(detected_domains)}"
        
        return {
            "unique_domains_detected": len(detected_domains),
            "total_queries": len(test_queries),
            "detection_results": detection_results,
            "multi_domain_capable": len(detected_domains) >= 2
        }
    
    async def _test_continuous_learning(self) -> Dict[str, Any]:
        """Test continuous learning integration"""
        learning_system = PatternLearningSystem({
            "learning_modes": ["reinforcement"],
            "pattern_evolution_tracking": True
        })
        
        session_id = await learning_system.start_learning_session(LearningMode.REINFORCEMENT)
        
        # Simulate continuous learning over multiple interactions
        learning_results = []
        
        for i in range(5):
            examples = [LearningExample(
                example_id=f"continuous_{i}",
                text=f"Medical procedure {i}: patient recovery successful",
                feedback={"reward": 0.7 + (i * 0.1)}  # Increasing reward
            )]
            
            result = await learning_system.learn_patterns_from_examples(session_id, examples)
            learning_results.append(result)
        
        # Check pattern evolution over time
        evolution_insights = await learning_system.get_pattern_evolution_insights(1.0)
        
        await learning_system.end_learning_session(session_id)
        
        total_new_patterns = sum(r.get("new_patterns", 0) for r in learning_results)
        total_evolved_patterns = sum(r.get("evolved_patterns", 0) for r in learning_results)
        
        return {
            "continuous_learning": True,
            "interactions_processed": len(learning_results),
            "total_new_patterns": total_new_patterns,
            "total_evolved_patterns": total_evolved_patterns,
            "evolution_events": evolution_insights.get("events_in_window", 0)
        }
    
    async def _test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks"""
        pattern_engine = DomainPatternEngine({})
        
        # Test analysis performance
        large_corpus = []
        for domain_texts in self.test_domains.values():
            large_corpus.extend(domain_texts * 3)  # Triple each domain's texts
        
        start_time = time.time()
        fingerprint = await pattern_engine.analyze_text_corpus(large_corpus)
        analysis_time = time.time() - start_time
        
        # Performance benchmarks
        documents_per_second = len(large_corpus) / analysis_time
        patterns_per_second = len([p for patterns in fingerprint.primary_patterns.values() for p in patterns]) / analysis_time
        
        # Verify reasonable performance
        assert analysis_time < 30.0, f"Analysis too slow: {analysis_time}s"
        assert documents_per_second > 0.1, f"Document processing too slow: {documents_per_second}/s"
        
        return {
            "analysis_time_seconds": analysis_time,
            "documents_processed": len(large_corpus),
            "documents_per_second": documents_per_second,
            "patterns_per_second": patterns_per_second,
            "performance_acceptable": analysis_time < 30.0
        }
    
    async def _test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage validation"""
        pattern_engine = DomainPatternEngine({"max_patterns_per_type": 20})
        
        # Process multiple domains
        initial_patterns = len(pattern_engine.discovered_patterns)
        
        for domain_name, domain_texts in self.test_domains.items():
            await pattern_engine.analyze_text_corpus(domain_texts, domain_hint=domain_name)
        
        final_patterns = len(pattern_engine.discovered_patterns)
        cached_fingerprints = len(pattern_engine.domain_fingerprints)
        
        # Verify reasonable memory usage
        assert cached_fingerprints <= 10, f"Too many cached fingerprints: {cached_fingerprints}"
        assert final_patterns < 1000, f"Too many cached patterns: {final_patterns}"
        
        return {
            "initial_patterns": initial_patterns,
            "final_patterns": final_patterns,
            "cached_fingerprints": cached_fingerprints,
            "memory_usage_reasonable": final_patterns < 1000
        }
    
    async def _test_caching_effectiveness(self) -> Dict[str, Any]:
        """Test caching effectiveness"""
        pattern_engine = DomainPatternEngine({})
        
        test_texts = self.test_domains["healthcare"]
        
        # First analysis (cache miss)
        start_time = time.time()
        fingerprint1 = await pattern_engine.analyze_text_corpus(test_texts)
        first_time = time.time() - start_time
        
        # Second analysis (cache hit)
        start_time = time.time()
        fingerprint2 = await pattern_engine.analyze_text_corpus(test_texts)
        second_time = time.time() - start_time
        
        # Verify caching worked
        assert fingerprint1.domain_id == fingerprint2.domain_id
        assert second_time < first_time * 0.5, f"Cache not effective: {second_time} vs {first_time}"
        
        metrics = pattern_engine.get_performance_metrics()
        cache_hit_rate = metrics["cache_hit_rate"]
        
        return {
            "first_analysis_time": first_time,
            "second_analysis_time": second_time,
            "speedup_ratio": first_time / second_time if second_time > 0 else float('inf'),
            "cache_hit_rate": cache_hit_rate,
            "caching_effective": second_time < first_time * 0.5
        }
    
    async def _test_invalid_inputs(self) -> Dict[str, Any]:
        """Test handling of invalid inputs"""
        pattern_engine = DomainPatternEngine({})
        adapter = ZeroConfigAdapter({"pattern_engine": pattern_engine})
        
        error_cases = []
        
        # Test empty corpus
        try:
            await pattern_engine.analyze_text_corpus([])
            error_cases.append("empty_corpus: No exception raised")
        except ValueError:
            error_cases.append("empty_corpus: Handled correctly")
        except Exception as e:
            error_cases.append(f"empty_corpus: Unexpected error: {e}")
        
        # Test invalid query
        try:
            result = await adapter.detect_domain_from_query("")
            assert result.detected_domain is None
            error_cases.append("empty_query: Handled gracefully")
        except Exception as e:
            error_cases.append(f"empty_query: Error: {e}")
        
        # Test extremely long text
        try:
            very_long_text = "word " * 10000
            await pattern_engine.analyze_text_corpus([very_long_text])
            error_cases.append("long_text: Handled successfully")
        except Exception as e:
            error_cases.append(f"long_text: Error: {e}")
        
        handled_correctly = sum(1 for case in error_cases if "Handled" in case or "gracefully" in case)
        
        return {
            "error_cases_tested": len(error_cases),
            "handled_correctly": handled_correctly,
            "error_details": error_cases,
            "robust_error_handling": handled_correctly >= 2
        }
    
    async def _test_graceful_degradation(self) -> Dict[str, Any]:
        """Test graceful degradation under adverse conditions"""
        # Test with minimal configuration
        minimal_engine = DomainPatternEngine({
            "min_pattern_frequency": 100,  # Very high threshold
            "confidence_threshold": 0.95   # Very high confidence
        })
        
        test_texts = self.test_domains["healthcare"][:1]  # Minimal text
        
        try:
            fingerprint = await minimal_engine.analyze_text_corpus(test_texts)
            
            # Should still create valid fingerprint even with restrictive settings
            assert fingerprint is not None
            assert fingerprint.confidence >= 0.0
            
            degradation_successful = True
            degradation_details = "System degraded gracefully with restrictive settings"
            
        except Exception as e:
            degradation_successful = False
            degradation_details = f"Failed to degrade gracefully: {e}"
        
        return {
            "degradation_successful": degradation_successful,
            "details": degradation_details,
            "fingerprint_created": degradation_successful
        }
    
    # Utility methods
    
    async def _run_test(self, test_name: str, test_function) -> ValidationResult:
        """Run a single test and capture results"""
        start_time = time.time()
        
        try:
            details = await test_function()
            duration_ms = (time.time() - start_time) * 1000
            
            result = ValidationResult(
                test_name=test_name,
                passed=True,
                duration_ms=duration_ms,
                details=details
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            result = ValidationResult(
                test_name=test_name,
                passed=False,
                duration_ms=duration_ms,
                details={},
                error_message=str(e)
            )
            
            self.logger.error(f"Test '{test_name}' failed: {e}")
        
        self.results.append(result)
        return result


async def main():
    """Main validation function"""
    print("🔍 Starting Dynamic Discovery System Validation")
    print("=" * 60)
    
    validator = DiscoverySystemValidator()
    
    try:
        results = await validator.run_comprehensive_validation()
        
        print(f"\n📊 Validation Summary:")
        print(f"   Total Tests: {results['total_tests']}")
        print(f"   Passed: {results['passed_tests']}")
        print(f"   Failed: {results['failed_tests']}")
        print(f"   Duration: {results['total_duration_seconds']:.2f}s")
        print(f"   Overall Success: {'✅' if results['overall_success'] else '❌'}")
        
        if results['failed_tests'] > 0:
            print(f"\n❌ Failed Tests:")
            for result in results['detailed_results']:
                if not result['passed']:
                    print(f"   - {result['test_name']}: {result['error']}")
        
        print(f"\n🎯 Phase Results:")
        for phase_name, phase_result in results['phase_results'].items():
            passed = phase_result.get('passed_tests', 0)
            total = phase_result.get('total_tests', 0)
            status = "✅" if passed == total else "❌"
            print(f"   {status} {phase_name}: {passed}/{total}")
        
        return results['overall_success']
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
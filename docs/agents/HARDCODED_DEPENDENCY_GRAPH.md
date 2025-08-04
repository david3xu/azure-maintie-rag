# 🕸️ Hardcoded Value Dependency Graph Analysis
============================================================

## 📊 Summary Statistics
- **Total Files Analyzed**: 48
- **Total Hardcoded Values Found**: 1279
- **Average per File**: 26.6

## 🎯 Hardcoded Value Clusters

### 🔸 THRESHOLDS (574 occurrences)
  - agents/__init__.py:71 - def __init__(self, success=True, result=None, execution_time=0.0):
  - agents/shared/capability_patterns.py:49 - self, key: str, value: Any, ttl: int = 3600, namespace: str = "default"
  - agents/shared/capability_patterns.py:151 - self, component: str, time_window_hours: int = 24
  - agents/shared/capability_patterns.py:218 - namespaced_key, redis_value, ttl=300
  - agents/shared/capability_patterns.py:231 - self, key: str, value: Any, ttl: int = 3600, namespace: str = "default"
  - agents/shared/capability_patterns.py:257 - invalidated_count = 0
  - agents/shared/capability_patterns.py:299 - if hit_rate > 80
  - agents/shared/capability_patterns.py:301 - if hit_rate > 60
  - agents/shared/capability_patterns.py:473 - std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
  - agents/shared/capability_patterns.py:473 - std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
  - ... and 564 more

### 🔸 ENTITY_LISTS (311 occurrences)
  - agents/__init__.py:228 - "agents": ["UniversalAgent", "DomainIntelligenceAgent"],
  - agents/__init__.py:235 - "tools": ["ConsolidatedToolManager"],
  - agents/__init__.py:236 - "search": ["TriModalOrchestrator"],
  - agents/shared/capability_patterns.py:202 - self.cache_stats["hits"] += 1
  - agents/shared/capability_patterns.py:203 - return entry["value"]
  - agents/shared/capability_patterns.py:213 - self.cache_stats["hits"] += 1
  - agents/shared/capability_patterns.py:214 - self.cache_stats["azure_redis_calls"] += 1
  - agents/shared/capability_patterns.py:227 - self.cache_stats["misses"] += 1
  - agents/shared/capability_patterns.py:245 - self.cache_stats["azure_redis_calls"] += 1
  - agents/shared/capability_patterns.py:265 - self.cache_stats["azure_redis_calls"] += 1
  - ... and 301 more

### 🔸 ENTITY_LISTS_PATTERNS (31 occurrences)
  - agents/__init__.py:234 - "intelligence": ["DomainAnalyzer", "PatternEngine"],
  - agents/shared/capability_patterns.py:422 - pattern_id=f"{pattern_type}_{pattern_data['id']}",
  - agents/shared/capability_patterns.py:423 - pattern_text=pattern_data["text"],
  - agents/shared/capability_patterns.py:425 - frequency=pattern_data["frequency"],
  - agents/shared/capability_patterns.py:427 - support=pattern_data["support"],
  - agents/shared/capability_patterns.py:923 - pattern_text=p["text"],
  - agents/knowledge_extraction/toolsets.py:332 - validation_results["recommendations"].append("Review relationship extraction patterns")
  - agents/knowledge_extraction/processors/validation_processor.py:958 - feedback["pattern_additions"].append(
  - agents/knowledge_extraction/processors/validation_processor.py:967 - feedback["pattern_additions"].append(
  - agents/knowledge_extraction/processors/validation_processor.py:1075 - metrics["relationship_pattern_coverage"] = len(
  - ... and 21 more

### 🔸 THRESHOLDS_CONFIDENCE (249 occurrences)
  - agents/shared/capability_patterns.py:67 - self, data: List[float], confidence_level: float = 0.95
  - agents/shared/capability_patterns.py:67 - self, data: List[float], confidence_level: float = 0.95
  - agents/shared/capability_patterns.py:329 - self, data: List[float], confidence_level: float = 0.95
  - agents/shared/capability_patterns.py:329 - self, data: List[float], confidence_level: float = 0.95
  - agents/shared/capability_patterns.py:459 - and pattern.confidence >= 0.6  # Minimum frequency
  - agents/shared/capability_patterns.py:459 - and pattern.confidence >= 0.6  # Minimum frequency
  - agents/shared/capability_patterns.py:930 - confidence_interval={"lower": 0.6, "upper": 0.9},
  - agents/shared/common_tools.py:115 - def validate_confidence_threshold(value: float, min_threshold: float = 0.0, max_threshold: float = 1.0) -> bool:
  - agents/shared/common_tools.py:115 - def validate_confidence_threshold(value: float, min_threshold: float = 0.0, max_threshold: float = 1.0) -> bool:
  - agents/shared/common_tools.py:115 - def validate_confidence_threshold(value: float, min_threshold: float = 0.0, max_threshold: float = 1.0) -> bool:
  - ... and 239 more

### 🔸 MAGIC_NUMBERS (63 occurrences)
  - agents/shared/capability_patterns.py:285 - return 0
  - agents/shared/capability_patterns.py:486 - return 0.0
  - agents/shared/capability_patterns.py:694 - optimized_config["top_k"] = 50  # Reduce to improve performance
  - agents/shared/capability_patterns.py:701 - base_cost = 0.1  # Base cost per job
  - agents/shared/common_tools.py:65 - return 0.0
  - agents/shared/common_tools.py:80 - return 0.0
  - agents/knowledge_extraction/toolsets.py:466 - return 0.0
  - agents/knowledge_extraction/toolsets.py:566 - return 0.0
  - agents/knowledge_extraction/toolsets.py:576 - return 0.0
  - agents/knowledge_extraction/toolsets.py:590 - return 1.0
  - ... and 53 more

### 🔸 ENTITY_LISTS_CONFIDENCE (26 occurrences)
  - agents/shared/capability_patterns.py:426 - confidence=pattern_data["confidence"],
  - agents/shared/capability_patterns.py:926 - confidence=p["confidence"],
  - agents/knowledge_extraction/toolsets.py:327 - validation_results["warnings"].append("Entity quality below threshold")
  - agents/knowledge_extraction/toolsets.py:328 - validation_results["recommendations"].append("Consider adjusting entity confidence threshold")
  - agents/knowledge_extraction/toolsets.py:331 - validation_results["warnings"].append("Relationship quality below threshold")
  - agents/knowledge_extraction/toolsets.py:524 - "confidence": self._calculate_cooccurrence_confidence(content, entity1["name"], entity2["name"]),
  - agents/knowledge_extraction/processors/validation_processor.py:304 - confidences = [e.get("confidence", 0.0) for e in entities]
  - agents/knowledge_extraction/processors/validation_processor.py:396 - confidences = [r.get("confidence", 0.0) for r in relationships]
  - agents/knowledge_extraction/processors/validation_processor.py:738 - confidences = [e.get("confidence", 0.0) for e in entities]
  - agents/knowledge_extraction/processors/validation_processor.py:782 - confidences = [r.get("confidence", 0.0) for r in relationships]
  - ... and 16 more

### 🔸 THRESHOLDS_PATTERNS (15 occurrences)
  - agents/shared/capability_patterns.py:457 - pattern.chi_square_p_value < 0.05
  - agents/shared/capability_patterns.py:458 - and pattern.frequency >= 3  # Statistically significant
  - agents/domain_intelligence/background_processor.py:57 - self.patterns_extracted: int = 0
  - agents/domain_intelligence/toolsets.py:55 - total_patterns = 0
  - agents/domain_intelligence/pattern_engine.py:85 - self, pattern_type: str, limit: int = 10
  - agents/domain_intelligence/pattern_engine.py:646 - self, domain: str, pattern_type: Optional[str] = None, limit: int = 50
  - agents/domain_intelligence/pattern_engine.py:687 - json.dump(serializable_patterns, f, indent=2)
  - agents/core/centralized_config.py:95 - pattern_frequency_minimum: int = 3
  - agents/core/cache_manager.py:65 - pattern_index_hits: int = 0
  - agents/core/cache_manager.py:168 - phrase_score = 2.0 * self.pattern_frequencies.get(phrase, 1)
  - ... and 5 more

### 🔸 MAGIC_NUMBERS_PATTERNS (5 occurrences)
  - agents/shared/capability_patterns.py:458 - and pattern.frequency >= 3  # Statistically significant
  - agents/domain_intelligence/toolsets.py:405 - return 5.0  # High diversity/patterns = more processing time
  - agents/domain_intelligence/toolsets.py:407 - return 3.5  # Medium diversity/patterns
  - agents/domain_intelligence/toolsets.py:409 - return 2.5  # Low diversity/patterns = faster processing
  - agents/domain_intelligence/config_generator.py:164 - learning_rate *= 0.5  # Lower learning rate for uncertain patterns

### 🔸 MAGIC_NUMBERS_CONFIDENCE (4 occurrences)
  - agents/shared/capability_patterns.py:459 - and pattern.confidence >= 0.6  # Minimum frequency
  - agents/knowledge_extraction/processors/relationship_processor.py:640 - base_confidence = 0.8  # Pattern-based matches are generally reliable
  - agents/knowledge_extraction/processors/entity_processor.py:270 - confidence = 0.8  # High confidence for known technical terms
  - agents/domain_intelligence/pattern_engine.py:82 - extraction_confidence: float = 0.8  # Overall extraction confidence score

## 🚨 Files with Most Hardcoded Values
- **agents/knowledge_extraction/processors/validation_processor.py**: 111 hardcoded values
  🔗 **Dependency Impact**: affects 2 levels
- **agents/interfaces/agent_contracts.py**: 105 hardcoded values
  🔗 **Dependency Impact**: affects 2 levels
- **agents/core/centralized_config.py**: 92 hardcoded values
  🔗 **Dependency Impact**: affects 1 levels
- **agents/knowledge_extraction/processors/relationship_processor.py**: 75 hardcoded values
  🔗 **Dependency Impact**: affects 4 levels
- **agents/shared/capability_patterns.py**: 65 hardcoded values
  🔗 **Dependency Impact**: affects 2 levels
- **agents/knowledge_extraction/processors/entity_processor.py**: 63 hardcoded values
  🔗 **Dependency Impact**: affects 5 levels
- **agents/knowledge_extraction/agent.py**: 57 hardcoded values
  🔗 **Dependency Impact**: affects 6 levels
- **agents/models/responses.py**: 54 hardcoded values
  🔗 **Dependency Impact**: affects 2 levels
- **agents/models/requests.py**: 54 hardcoded values
  🔗 **Dependency Impact**: affects 2 levels
- **agents/domain_intelligence/statistical_domain_analyzer.py**: 53 hardcoded values
  🔗 **Dependency Impact**: affects 1 levels

## 🧠 Domain Intelligence Agent Analysis

### 📁 agents/domain_intelligence/hybrid_domain_analyzer.py
  - Line 85: `processing_time: float = 0.0`
    🏷️ Type: thresholds, Value: float = 0.0
  - Line 104: `max_features=1000,`
    🏷️ Type: thresholds, Value: max_features=1000
  - Line 107: `min_df=0.01,`
    🏷️ Type: thresholds, Value: min_df=0.01
  - Line 108: `max_df=0.95,`
    🏷️ Type: thresholds, Value: max_df=0.95
  - Line 115: `self.analysis_count = 0`
    🏷️ Type: thresholds, Value: analysis_count = 0
  - Line 116: `self.llm_calls = 0`
    🏷️ Type: thresholds, Value: llm_calls = 0
  - Line 117: `self.total_processing_time = 0.0`
    🏷️ Type: thresholds, Value: total_processing_time = 0.0
  - Line 229: `temperature=0.1,  # Low temperature for consistent results`
    🏷️ Type: thresholds, Value: temperature=0.1
  - Line 230: `max_tokens=1000,`
    🏷️ Type: thresholds, Value: max_tokens=1000
  - Line 256: `"semantic_relationships": [["entity1", "relationship", "entity2"], ...],`
    🏷️ Type: entity_lists, Value: [["entity1", "relationship", "entity2"], ...]
  - Line 332: `recommended_strategies=["general_extraction"],`
    🏷️ Type: entity_lists, Value: ["general_extraction"]
  - Line 405: `base_size = 1000  # Default chunk size`
    🏷️ Type: thresholds, Value: base_size = 1000
  - Line 405: `base_size = 1000  # Default chunk size`
    🏷️ Type: magic_numbers, Value: = 1000  #
  - Line 416: `if entity_density > 20:  # High entity density`
    🏷️ Type: thresholds, Value: entity_density > 20
  - Line 418: `elif entity_density < 5:  # Low entity density`
    🏷️ Type: thresholds, Value: entity_density < 5
  - Line 423: `complexity_multiplier *= 0.8  # Smaller chunks for technical specs`
    🏷️ Type: magic_numbers, Value: = 0.8  #
  - Line 432: `base_ratio = 0.2  # 20% default overlap`
    🏷️ Type: thresholds, Value: base_ratio = 0.2
  - Line 432: `base_ratio = 0.2  # 20% default overlap`
    🏷️ Type: magic_numbers, Value: = 0.2  #
  - Line 436: `base_ratio = 0.25`
    🏷️ Type: thresholds, Value: base_ratio = 0.25
  - Line 440: `base_ratio = 0.3`
    🏷️ Type: thresholds, Value: base_ratio = 0.3
  - Line 444: `base_ratio = 0.15`
    🏷️ Type: thresholds, Value: base_ratio = 0.15
  - Line 480: `entity_factor = 1.0 + (len(llm_extraction.key_entities) / 100)`
    🏷️ Type: thresholds, Value: entity_factor = 1.0
  - Line 483: `relationship_factor = 1.0 + (len(llm_extraction.semantic_relationships) / 50)`
    🏷️ Type: thresholds, Value: relationship_factor = 1.0
  - Line 534: `timeout_base = 30  # seconds`
    🏷️ Type: thresholds, Value: timeout_base = 30
  - Line 534: `timeout_base = 30  # seconds`
    🏷️ Type: magic_numbers, Value: = 30  #
  - Line 565: `llm_confidence = {"high": 0.9, "medium": 0.7, "low": 0.5}.get(`
    🏷️ Type: thresholds, Value: confidence = {"high": 0.9, "medium": 0.7, "low": 0.5
  - Line 580: `hybrid_confidence = llm_confidence * 0.6 + stat_confidence * 0.4`
    🏷️ Type: thresholds, Value: confidence = llm_confidence * 0.6 + stat_confidence * 0.4
  - Line 684: `optimal_chunk_size=1000,`
    🏷️ Type: thresholds, Value: optimal_chunk_size=1000
  - Line 685: `chunk_overlap_ratio=0.2,`
    🏷️ Type: thresholds, Value: chunk_overlap_ratio=0.2
  - Line 686: `entity_density=0.0,`
    🏷️ Type: thresholds, Value: entity_density=0.0
  - Line 687: `relationship_density=0.0,`
    🏷️ Type: thresholds, Value: relationship_density=0.0
  - Line 688: `vocabulary_complexity=0.0,`
    🏷️ Type: thresholds, Value: vocabulary_complexity=0.0
  - Line 689: `processing_load_estimate=0.0,`
    🏷️ Type: thresholds, Value: processing_load_estimate=0.0
  - Line 700: `hybrid_confidence=0.1,`
    🏷️ Type: thresholds, Value: hybrid_confidence=0.1
  - Line 700: `hybrid_confidence=0.1,`
    🏷️ Type: thresholds, Value: confidence=0.1
  - Line 740: `return optimizations.get(domain_type, optimizations["general"])`
    🏷️ Type: entity_lists, Value: ["general"]

### 📁 agents/domain_intelligence/statistical_domain_analyzer.py
  - Line 48: `processing_time: float = 0.0`
    🏷️ Type: thresholds, Value: float = 0.0
  - Line 81: `max_features=1000,`
    🏷️ Type: thresholds, Value: max_features=1000
  - Line 84: `min_df=0.01,`
    🏷️ Type: thresholds, Value: min_df=0.01
  - Line 85: `max_df=0.95,`
    🏷️ Type: thresholds, Value: max_df=0.95
  - Line 89: `self.clusterer = KMeans(n_clusters=5, random_state=42, n_init=10)`
    🏷️ Type: thresholds, Value: n_clusters=5
  - Line 89: `self.clusterer = KMeans(n_clusters=5, random_state=42, n_init=10)`
    🏷️ Type: thresholds, Value: random_state=42
  - Line 89: `self.clusterer = KMeans(n_clusters=5, random_state=42, n_init=10)`
    🏷️ Type: thresholds, Value: n_init=10
  - Line 96: `self.analysis_count = 0`
    🏷️ Type: thresholds, Value: analysis_count = 0
  - Line 97: `self.total_processing_time = 0.0`
    🏷️ Type: thresholds, Value: total_processing_time = 0.0
  - Line 268: `return 0.0`
    🏷️ Type: magic_numbers, Value: return 0.0
  - Line 271: `entropy = -sum(f * math.log2(f) for f in frequencies if f > 0)`
    🏷️ Type: thresholds, Value: f > 0
  - Line 289: `math.log(unique_words) / math.log(total_words) if total_words > 1 else 0.0`
    🏷️ Type: thresholds, Value: total_words > 1
  - Line 330: `if score > 0:`
    🏷️ Type: thresholds, Value: score > 0
  - Line 369: `signatures["top_term_concentration"] = sum(freq for _, freq in top_terms)`
    🏷️ Type: entity_lists, Value: ["top_term_concentration"]
  - Line 370: `signatures["frequency_distribution_skew"] = self._calculate_frequency_skew(`
    🏷️ Type: entity_lists, Value: ["frequency_distribution_skew"]
  - Line 387: `intervals["term_frequency"] = (`
    🏷️ Type: entity_lists, Value: ["term_frequency"]
  - Line 412: `hypotheses["technical"] = {`
    🏷️ Type: entity_lists, Value: ["technical"]
  - Line 414: `"features": ["lexical_diversity", "entropy", "word_length"],`
    🏷️ Type: entity_lists, Value: ["lexical_diversity", "entropy", "word_length"]
  - Line 424: `hypotheses["process"] = {`
    🏷️ Type: entity_lists, Value: ["process"]
  - Line 439: `hypotheses["academic"] = {`
    🏷️ Type: entity_lists, Value: ["academic"]
  - Line 441: `"features": ["entropy", "lexical_diversity", "sentence_length"],`
    🏷️ Type: entity_lists, Value: ["entropy", "lexical_diversity", "sentence_length"]
  - Line 445: `general_score = 0.5  # Baseline score`
    🏷️ Type: thresholds, Value: general_score = 0.5
  - Line 445: `general_score = 0.5  # Baseline score`
    🏷️ Type: magic_numbers, Value: = 0.5  #
  - Line 446: `hypotheses["general"] = {"score": general_score, "features": ["baseline"]}`
    🏷️ Type: entity_lists, Value: ["general"] = {"score": general_score, "features": ["baseline"]
  - Line 478: `return 0.0`
    🏷️ Type: magic_numbers, Value: return 0.0
  - Line 482: `if n == 0:`
    🏷️ Type: thresholds, Value: n == 0
  - Line 483: `return 0.0`
    🏷️ Type: magic_numbers, Value: return 0.0
  - Line 486: `cumsum = 0`
    🏷️ Type: thresholds, Value: cumsum = 0
  - Line 495: `return 0.0`
    🏷️ Type: magic_numbers, Value: return 0.0
  - Line 501: `if std_freq == 0:`
    🏷️ Type: thresholds, Value: std_freq == 0
  - Line 502: `return 0.0`
    🏷️ Type: magic_numbers, Value: return 0.0
  - Line 517: `if analysis.entropy_score > 3.0:`
    🏷️ Type: thresholds, Value: entropy_score > 3.0
  - Line 523: `if analysis.vocabulary_richness > 0.6:`
    🏷️ Type: thresholds, Value: vocabulary_richness > 0.6
  - Line 532: `if avg_complexity > 0.5:`
    🏷️ Type: thresholds, Value: avg_complexity > 0.5
  - Line 538: `if analysis.word_count > 1000:`
    🏷️ Type: thresholds, Value: word_count > 1000
  - Line 572: `high_importance = sum(1 for score in tfidf_scores.values() if score > 0.1)`
    🏷️ Type: thresholds, Value: score > 0.1
  - Line 574: `1 for score in tfidf_scores.values() if 0.05 <= score <= 0.1`
    🏷️ Type: thresholds, Value: score <= 0.1
  - Line 579: `if total_features == 0:`
    🏷️ Type: thresholds, Value: total_features == 0
  - Line 603: `if entropy < 2.0:`
    🏷️ Type: thresholds, Value: entropy < 2.0
  - Line 604: `return 0.2  # Low entropy`
    🏷️ Type: magic_numbers, Value: return 0.2
  - Line 605: `elif entropy < 4.0:`
    🏷️ Type: thresholds, Value: entropy < 4.0
  - Line 606: `return 0.5  # Medium entropy`
    🏷️ Type: magic_numbers, Value: return 0.5
  - Line 607: `elif entropy < 6.0:`
    🏷️ Type: thresholds, Value: entropy < 6.0
  - Line 608: `return 0.8  # High entropy`
    🏷️ Type: magic_numbers, Value: return 0.8
  - Line 610: `return 1.0  # Very high entropy`
    🏷️ Type: magic_numbers, Value: return 1.0
  - Line 617: `alpha = 0.05  # Standard significance level`
    🏷️ Type: thresholds, Value: alpha = 0.05
  - Line 617: `alpha = 0.05  # Standard significance level`
    🏷️ Type: magic_numbers, Value: = 0.05  #
  - Line 621: `p_value = 1.0 - confidence`
    🏷️ Type: thresholds, Value: p_value = 1.0
  - Line 630: `word_count=0,`
    🏷️ Type: thresholds, Value: word_count=0
  - Line 631: `unique_words=0,`
    🏷️ Type: thresholds, Value: unique_words=0
  - Line 632: `avg_sentence_length=0.0,`
    🏷️ Type: thresholds, Value: avg_sentence_length=0.0
  - Line 633: `vocabulary_richness=0.0,`
    🏷️ Type: thresholds, Value: vocabulary_richness=0.0
  - Line 635: `entropy_score=0.0,`
    🏷️ Type: thresholds, Value: entropy_score=0.0

### 📁 agents/domain_intelligence/background_processor.py
  - Line 12: `- Achieve >95% cache hit rates for domain detection`
    🏷️ Type: thresholds, Value: Achieve >95
  - Line 53: `self.start_time: float = 0.0`
    🏷️ Type: thresholds, Value: float = 0.0
  - Line 54: `self.end_time: float = 0.0`
    🏷️ Type: thresholds, Value: float = 0.0
  - Line 55: `self.domains_processed: int = 0`
    🏷️ Type: thresholds, Value: int = 0
  - Line 56: `self.files_processed: int = 0`
    🏷️ Type: thresholds, Value: int = 0
  - Line 57: `self.patterns_extracted: int = 0`
    🏷️ Type: thresholds, Value: int = 0
  - Line 58: `self.configurations_generated: int = 0`
    🏷️ Type: thresholds, Value: int = 0
  - Line 59: `self.cache_entries_created: int = 0`
    🏷️ Type: thresholds, Value: int = 0
  - Line 64: `return self.end_time - self.start_time if self.end_time > 0 else 0.0`
    🏷️ Type: thresholds, Value: end_time > 0
  - Line 68: `return self.files_processed / self.total_time if self.total_time > 0 else 0.0`
    🏷️ Type: thresholds, Value: total_time > 0
  - Line 102: `self.thread_pool = ThreadPoolExecutor(max_workers=4)`
    🏷️ Type: thresholds, Value: max_workers=4
  - Line 147: `f"✅ Background processing complete in {stats_dict['total_time']:.2f}s"`
    🏷️ Type: entity_lists, Value: ['total_time']
  - Line 150: `f"📊 Processed {stats_dict['domains_processed']} domains, "`
    🏷️ Type: entity_lists, Value: ['domains_processed']
  - Line 151: `f"{stats_dict['files_processed']} files, "`
    🏷️ Type: entity_lists, Value: ['files_processed']
  - Line 152: `f"extracted {stats_dict['patterns_extracted']} patterns"`
    🏷️ Type: entity_lists, Value: ['patterns_extracted']
  - Line 155: `f"⚡ Processing rate: {stats_dict['files_per_second']:.1f} files/sec"`
    🏷️ Type: entity_lists, Value: ['files_per_second']
  - Line 157: `logger.info(f"🎯 Success rate: {stats_dict['success_rate']*100:.1f}%")`
    🏷️ Type: entity_lists, Value: ['success_rate']
  - Line 211: `files_processed = 0`
    🏷️ Type: thresholds, Value: files_processed = 0
  - Line 251: `self.stats.cache_entries_created += 2  # signature + config`
    🏷️ Type: magic_numbers, Value: = 2  #
  - Line 306: `total_word_count = 0`
    🏷️ Type: thresholds, Value: total_word_count = 0
  - Line 327: `avg_confidence = 0.3`
    🏷️ Type: thresholds, Value: avg_confidence = 0.3
  - Line 327: `avg_confidence = 0.3`
    🏷️ Type: thresholds, Value: confidence = 0.3
  - Line 331: `p.pattern_text for p in merged_entities if p.confidence > 0.8`
    🏷️ Type: thresholds, Value: confidence > 0.8
  - Line 371: `f"✅ Pattern indexes built: {index_stats['pattern_index_size']} patterns, "`
    🏷️ Type: entity_lists, Value: ['pattern_index_size']
  - Line 372: `f"{index_stats['total_domains']} domains"`
    🏷️ Type: entity_lists, Value: ['total_domains']
  - Line 386: `f"✅ Cache optimized: {cache_stats['active_entries']} active entries, "`
    🏷️ Type: entity_lists, Value: ['active_entries']
  - Line 388: `f"{cache_stats['pattern_index_size']} pattern indexes ready"`
    🏷️ Type: entity_lists, Value: ['pattern_index_size']
  - Line 394: `"is_processing": self.stats.start_time > 0 and self.stats.end_time == 0,`
    🏷️ Type: thresholds, Value: start_time > 0
  - Line 394: `"is_processing": self.stats.start_time > 0 and self.stats.end_time == 0,`
    🏷️ Type: thresholds, Value: end_time == 0

### 📁 agents/domain_intelligence/toolsets.py
  - Line 55: `total_patterns = 0`
    🏷️ Type: thresholds, Value: total_patterns = 0
  - Line 66: `if file_count > 0:`
    🏷️ Type: thresholds, Value: file_count > 0
  - Line 93: `total_tokens = 0`
    🏷️ Type: thresholds, Value: total_tokens = 0
  - Line 94: `total_chars = 0`
    🏷️ Type: thresholds, Value: total_chars = 0
  - Line 327: `if avg_doc_length > 2000:`
    🏷️ Type: thresholds, Value: avg_doc_length > 2000
  - Line 329: `elif avg_doc_length > 800:`
    🏷️ Type: thresholds, Value: avg_doc_length > 800
  - Line 359: `semantic_clusters['syntax_patterns'].append(token)`
    🏷️ Type: entity_lists, Value: ['syntax_patterns']
  - Line 363: `semantic_clusters['compound_terms'].append(token)`
    🏷️ Type: entity_lists, Value: ['compound_terms']
  - Line 367: `semantic_clusters['technical_abbreviations'].append(token)`
    🏷️ Type: entity_lists, Value: ['technical_abbreviations']
  - Line 371: `semantic_clusters['domain_terms'].append(token)`
    🏷️ Type: entity_lists, Value: ['domain_terms']
  - Line 374: `code_patterns = semantic_clusters['syntax_patterns'][:extraction_config.code_elements_limit]`
    🏷️ Type: entity_lists, Value: ['syntax_patterns'][:extraction_config.code_elements_limit]
  - Line 375: `api_patterns = semantic_clusters['technical_abbreviations'][:extraction_config.api_interfaces_limit]`
    🏷️ Type: entity_lists, Value: ['technical_abbreviations'][:extraction_config.api_interfaces_limit]
  - Line 376: `data_patterns = semantic_clusters['compound_terms'][:extraction_config.data_structures_limit]`
    🏷️ Type: entity_lists, Value: ['compound_terms'][:extraction_config.data_structures_limit]
  - Line 379: `rules['code_elements'] = code_patterns[:extraction_config.code_elements_limit]`
    🏷️ Type: entity_lists, Value: ['code_elements'] = code_patterns[:extraction_config.code_elements_limit]
  - Line 381: `rules['api_interfaces'] = api_patterns[:extraction_config.api_interfaces_limit]`
    🏷️ Type: entity_lists, Value: ['api_interfaces'] = api_patterns[:extraction_config.api_interfaces_limit]
  - Line 383: `rules['data_structures'] = data_patterns[:extraction_config.data_structures_limit]`
    🏷️ Type: entity_lists, Value: ['data_structures'] = data_patterns[:extraction_config.data_structures_limit]
  - Line 387: `rules['general_concepts'] = top_tokens[:extraction_config.general_concepts_limit]`
    🏷️ Type: entity_lists, Value: ['general_concepts'] = top_tokens[:extraction_config.general_concepts_limit]
  - Line 404: `if complexity_score > 1.5:`
    🏷️ Type: thresholds, Value: complexity_score > 1.5
  - Line 405: `return 5.0  # High diversity/patterns = more processing time`
    🏷️ Type: magic_numbers, Value: return 5.0
  - Line 406: `elif complexity_score > 0.8:`
    🏷️ Type: thresholds, Value: complexity_score > 0.8
  - Line 407: `return 3.5  # Medium diversity/patterns`
    🏷️ Type: magic_numbers, Value: return 3.5
  - Line 409: `return 2.5  # Low diversity/patterns = faster processing`
    🏷️ Type: magic_numbers, Value: return 2.5

### 📁 agents/domain_intelligence/agent.py
  - Line 22: `confidence: float = Field(description="Confidence score (0.0-1.0)")`
    🏷️ Type: thresholds, Value: confidence: float = Field(description="Confidence score (0.0-1.0

### 📁 agents/domain_intelligence/__init__.py
  - Line 21: `- Background processing for <5ms domain detection`
    🏷️ Type: thresholds, Value: for <5

### 📁 agents/domain_intelligence/pattern_engine.py
  - Line 44: `usage_count: int = 0`
    🏷️ Type: thresholds, Value: int = 0
  - Line 56: `base_score *= 1.5  # Boost for domain match`
    🏷️ Type: magic_numbers, Value: = 1.5  #
  - Line 65: `"""Check if pattern has high confidence (>0.7 and frequency >2)"""`
    🏷️ Type: thresholds, Value: frequency >2
  - Line 66: `return self.confidence > 0.7 and self.frequency > 2`
    🏷️ Type: thresholds, Value: confidence > 0.7
  - Line 66: `return self.confidence > 0.7 and self.frequency > 2`
    🏷️ Type: thresholds, Value: frequency > 2
  - Line 80: `source_word_count: int = 0`
    🏷️ Type: thresholds, Value: int = 0
  - Line 81: `processing_time: float = 0.0`
    🏷️ Type: thresholds, Value: float = 0.0
  - Line 82: `extraction_confidence: float = 0.8  # Overall extraction confidence score`
    🏷️ Type: thresholds, Value: float = 0.8
  - Line 82: `extraction_confidence: float = 0.8  # Overall extraction confidence score`
    🏷️ Type: thresholds, Value: confidence: float = 0.8
  - Line 82: `extraction_confidence: float = 0.8  # Overall extraction confidence score`
    🏷️ Type: magic_numbers, Value: = 0.8  #
  - Line 85: `self, pattern_type: str, limit: int = 10`
    🏷️ Type: thresholds, Value: int = 10
  - Line 133: `self.min_confidence_threshold = self.config.get("min_confidence_threshold", 0.3)`
    🏷️ Type: thresholds, Value: confidence_threshold = self.config.get("min_confidence_threshold", 0.3
  - Line 203: `self.stats["total_extractions"] += 1`
    🏷️ Type: entity_lists, Value: ["total_extractions"]
  - Line 204: `self.stats["avg_extraction_time"] = (`
    🏷️ Type: entity_lists, Value: ["avg_extraction_time"]
  - Line 205: `self.stats["avg_extraction_time"]`
    🏷️ Type: entity_lists, Value: ["avg_extraction_time"]
  - Line 206: `* (self.stats["total_extractions"] - 1)`
    🏷️ Type: entity_lists, Value: ["total_extractions"]
  - Line 208: `) / self.stats["total_extractions"]`
    🏷️ Type: entity_lists, Value: ["total_extractions"]
  - Line 236: `source_word_count=0,`
    🏷️ Type: thresholds, Value: source_word_count=0
  - Line 318: `confidence=confidence * 0.8,`
    🏷️ Type: thresholds, Value: confidence=confidence * 0.8
  - Line 405: `confidence=confidence * 0.6,`
    🏷️ Type: thresholds, Value: confidence=confidence * 0.6
  - Line 458: `if overlap > 0.3:  # 30% word overlap threshold`
    🏷️ Type: thresholds, Value: overlap > 0.3
  - Line 459: `cluster["patterns"].append(other_pattern.pattern_text)`
    🏷️ Type: entity_lists, Value: ["patterns"]
  - Line 460: `cluster["confidence"] = max(`
    🏷️ Type: entity_lists, Value: ["confidence"]
  - Line 461: `cluster["confidence"], other_pattern.confidence`
    🏷️ Type: entity_lists, Value: ["confidence"]
  - Line 463: `cluster["pattern_count"] += 1`
    🏷️ Type: entity_lists, Value: ["pattern_count"]
  - Line 470: `clusters.sort(key=lambda c: (c["pattern_count"], c["confidence"]), reverse=True)`
    🏷️ Type: entity_lists, Value: ["pattern_count"], c["confidence"]
  - Line 519: `self.stats["patterns_learned"] += 1`
    🏷️ Type: entity_lists, Value: ["patterns_learned"]
  - Line 599: `self, query: str, domain: Optional[str] = None, limit: int = 20`
    🏷️ Type: thresholds, Value: int = 20
  - Line 641: `self.stats["patterns_applied"] += 1`
    🏷️ Type: entity_lists, Value: ["patterns_applied"]
  - Line 646: `self, domain: str, pattern_type: Optional[str] = None, limit: int = 50`
    🏷️ Type: thresholds, Value: int = 50
  - Line 687: `json.dump(serializable_patterns, f, indent=2)`
    🏷️ Type: thresholds, Value: indent=2
  - Line 734: `for pattern_type in ["entity", "action", "relationship", "temporal"]`
    🏷️ Type: entity_lists, Value: ["entity", "action", "relationship", "temporal"]

### 📁 agents/domain_intelligence/config_generator.py
  - Line 133: `p.pattern_text for p in patterns.entity_patterns if p.confidence > 0.6`
    🏷️ Type: thresholds, Value: confidence > 0.6
  - Line 136: `p.pattern_text for p in patterns.relationship_patterns if p.confidence > 0.5`
    🏷️ Type: thresholds, Value: confidence > 0.5
  - Line 147: `inferred_relationships if inferred_relationships else ["connects"]`
    🏷️ Type: entity_lists, Value: ["connects"]
  - Line 162: `learning_rate = base_config["learning_rate"]`
    🏷️ Type: entity_lists, Value: ["learning_rate"]
  - Line 163: `if avg_confidence < 0.6:`
    🏷️ Type: thresholds, Value: avg_confidence < 0.6
  - Line 164: `learning_rate *= 0.5  # Lower learning rate for uncertain patterns`
    🏷️ Type: magic_numbers, Value: = 0.5  #
  - Line 206: `complexity_score = 0`
    🏷️ Type: thresholds, Value: complexity_score = 0
  - Line 208: `if entity_count > 100:`
    🏷️ Type: thresholds, Value: entity_count > 100
  - Line 210: `elif entity_count > 50:`
    🏷️ Type: thresholds, Value: entity_count > 50
  - Line 213: `if high_confidence_count > 20:`
    🏷️ Type: thresholds, Value: high_confidence_count > 20
  - Line 215: `elif high_confidence_count > 10:`
    🏷️ Type: thresholds, Value: high_confidence_count > 10
  - Line 218: `if relationship_count > 5:`
    🏷️ Type: thresholds, Value: relationship_count > 5
  - Line 221: `if patterns.extraction_confidence > 0.8:`
    🏷️ Type: thresholds, Value: extraction_confidence > 0.8
  - Line 224: `if complexity_score >= 4:`
    🏷️ Type: thresholds, Value: complexity_score >= 4
  - Line 226: `elif complexity_score >= 2:`
    🏷️ Type: thresholds, Value: complexity_score >= 2
  - Line 264: `if any(verb in text for verb in ["connect", "link", "join", "bind"]):`
    🏷️ Type: entity_lists, Value: ["connect", "link", "join", "bind"]
  - Line 266: `if any(verb in text for verb in ["contain", "include", "hold", "have"]):`
    🏷️ Type: entity_lists, Value: ["contain", "include", "hold", "have"]
  - Line 268: `if any(verb in text for verb in ["use", "utiliz", "employ", "apply"]):`
    🏷️ Type: entity_lists, Value: ["use", "utiliz", "employ", "apply"]
  - Line 270: `if any(verb in text for verb in ["create", "generat", "produc", "make"]):`
    🏷️ Type: entity_lists, Value: ["create", "generat", "produc", "make"]
  - Line 272: `if any(verb in text for verb in ["part", "component", "element", "piece"]):`
    🏷️ Type: entity_lists, Value: ["part", "component", "element", "piece"]
  - Line 274: `if any(verb in text for verb in ["depend", "rel", "requir", "need"]):`
    🏷️ Type: entity_lists, Value: ["depend", "rel", "requir", "need"]

### 📁 agents/domain_intelligence/domain_analyzer.py
  - Line 43: `processing_time: float = 0.0`
    🏷️ Type: thresholds, Value: float = 0.0
  - Line 165: `self.analysis_stats["total_analyses"] += 1`
    🏷️ Type: entity_lists, Value: ["total_analyses"]
  - Line 166: `self.analysis_stats["avg_processing_time"] = (`
    🏷️ Type: entity_lists, Value: ["avg_processing_time"]
  - Line 167: `self.analysis_stats["avg_processing_time"]`
    🏷️ Type: entity_lists, Value: ["avg_processing_time"]
  - Line 168: `* (self.analysis_stats["total_analyses"] - 1)`
    🏷️ Type: entity_lists, Value: ["total_analyses"]
  - Line 170: `) / self.analysis_stats["total_analyses"]`
    🏷️ Type: entity_lists, Value: ["total_analyses"]
  - Line 208: `domain_scores[user_domain] *= 1.5  # Boost user-specified domain`
    🏷️ Type: magic_numbers, Value: = 1.5  #
  - Line 221: `confidence = min(1.0, primary_score / max(1.0, sum(domain_scores.values())))`
    🏷️ Type: thresholds, Value: confidence = min(1.0, primary_score / max(1.0
  - Line 396: `indicators["concept_density"] = concept_density`
    🏷️ Type: entity_lists, Value: ["concept_density"]
  - Line 397: `indicators["vocabulary_richness"] = (`
    🏷️ Type: entity_lists, Value: ["vocabulary_richness"]
  - Line 398: `len(concept_frequency) / total_concepts if total_concepts > 0 else 0`
    🏷️ Type: thresholds, Value: total_concepts > 0
  - Line 417: `return 0.0`
    🏷️ Type: magic_numbers, Value: return 0.0
  - Line 453: `return 0.0`
    🏷️ Type: magic_numbers, Value: return 0.0
  - Line 456: `technical_indicators = 0`
    🏷️ Type: thresholds, Value: technical_indicators = 0
  - Line 508: `scores["general"] = 1.0`
    🏷️ Type: entity_lists, Value: ["general"]
  - Line 547: `if analysis.technical_density > 0.1:`
    🏷️ Type: thresholds, Value: technical_density > 0.1
  - Line 552: `if analysis.complexity_score > 0.7:`
    🏷️ Type: thresholds, Value: complexity_score > 0.7
  - Line 569: `scores["technical_content"] += 1.0`
    🏷️ Type: entity_lists, Value: ["technical_content"]
  - Line 571: `scores["procedural_content"] += 1.0`
    🏷️ Type: entity_lists, Value: ["procedural_content"]
  - Line 573: `scores["academic_content"] += 1.0`
    🏷️ Type: entity_lists, Value: ["academic_content"]
  - Line 589: `scores["configuration_domain"] += 1.0`
    🏷️ Type: entity_lists, Value: ["configuration_domain"]
  - Line 591: `scores["maintenance_domain"] += 1.0`
    🏷️ Type: entity_lists, Value: ["maintenance_domain"]
  - Line 593: `scores["analytical_domain"] += 1.0`
    🏷️ Type: entity_lists, Value: ["analytical_domain"]
  - Line 604: `if analysis.technical_density > 0.3:`
    🏷️ Type: thresholds, Value: technical_density > 0.3
  - Line 605: `scores["high_technical_density"] += analysis.technical_density * 2`
    🏷️ Type: entity_lists, Value: ["high_technical_density"]
  - Line 607: `if analysis.complexity_score > 0.7:`
    🏷️ Type: thresholds, Value: complexity_score > 0.7
  - Line 608: `scores["complex_content"] += analysis.complexity_score * 1.5`
    🏷️ Type: entity_lists, Value: ["complex_content"]
  - Line 610: `if analysis.vocabulary_richness > 0.5:`
    🏷️ Type: thresholds, Value: vocabulary_richness > 0.5
  - Line 611: `scores["rich_vocabulary"] += analysis.vocabulary_richness * 1.2`
    🏷️ Type: entity_lists, Value: ["rich_vocabulary"]
  - Line 615: `scores["concept_rich"] += len(analysis.concept_frequency) / 50`
    🏷️ Type: entity_lists, Value: ["concept_rich"]
  - Line 734: `confidence=0.3,`
    🏷️ Type: thresholds, Value: confidence=0.3
  - Line 734: `confidence=0.3,`
    🏷️ Type: thresholds, Value: confidence=0.3
  - Line 832: `"total_analyses": self.analysis_stats["total_analyses"],`
    🏷️ Type: entity_lists, Value: ["total_analyses"]
  - Line 833: `"avg_processing_time": self.analysis_stats["avg_processing_time"],`
    🏷️ Type: entity_lists, Value: ["avg_processing_time"]
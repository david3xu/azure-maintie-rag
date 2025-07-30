#!/usr/bin/env python3
"""
Domain Pattern Alignment Analysis
Compare knowledge extraction results with predetermined domain patterns
"""

import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Set
import argparse

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.domain_patterns import DomainPatternManager, MAINTENANCE_PATTERNS

class DomainPatternAlignmentAnalyzer:
    """Analyze how well extraction results align with domain patterns"""
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.data = self._load_results()
        self.domain = self.data.get('domain', 'maintenance')
        
    def _load_results(self) -> Dict[str, Any]:
        """Load knowledge extraction results"""
        try:
            with open(self.results_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Results file not found: {self.results_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in results file: {e}")
            sys.exit(1)
    
    def analyze_alignment(self):
        """Run comprehensive alignment analysis"""
        print("🔍 Domain Pattern Alignment Analysis")
        print("=" * 50)
        
        # Get domain patterns
        patterns = DomainPatternManager.get_patterns(self.domain)
        prompts = DomainPatternManager.get_prompts(self.domain)
        metadata = DomainPatternManager.get_metadata(self.domain)
        
        print(f"📋 Domain: {self.domain}")
        print(f"🎯 Extraction Focus: {prompts.extraction_focus}")
        print()
        
        # Analyze entity alignment
        self._analyze_entity_alignment(patterns, prompts)
        print()
        
        # Analyze relationship alignment  
        self._analyze_relationship_alignment(metadata)
        print()
        
        # Analyze terminology alignment
        self._analyze_terminology_alignment(patterns)
        print()
        
        # Overall alignment score
        self._calculate_alignment_score(patterns, prompts, metadata)
        
    def _analyze_entity_alignment(self, patterns, prompts):
        """Analyze how extracted entities align with expected types"""
        entities = self.data.get('knowledge_data', {}).get('entities', [])
        
        print("🏷️  Entity Type Alignment")
        print("-" * 25)
        
        # Expected entity types from extraction focus
        expected_types = set(prompts.extraction_focus.split(', '))
        print(f"Expected Types (from prompt): {', '.join(sorted(expected_types))}")
        
        # Actual entity types
        actual_types = Counter(entity.get('type', 'unknown') for entity in entities)
        print(f"Actual Types Found: {', '.join(sorted(actual_types.keys()))}")
        
        # Calculate alignment
        expected_normalized = {t.strip().lower() for t in expected_types}
        actual_normalized = {t.lower() for t in actual_types.keys()}
        
        # Direct matches
        direct_matches = expected_normalized.intersection(actual_normalized)
        
        # Partial matches (plural/singular, contains)
        partial_matches = set()
        for expected in expected_normalized:
            for actual in actual_normalized:
                if expected in actual or actual in expected:
                    if expected not in direct_matches and actual not in direct_matches:
                        partial_matches.add((expected, actual))
        
        print(f"\n✅ Direct Matches: {len(direct_matches)}")
        for match in sorted(direct_matches):
            print(f"  - {match}")
            
        print(f"\n🔄 Partial Matches: {len(partial_matches)}")
        for expected, actual in sorted(partial_matches):
            print(f"  - '{expected}' ↔ '{actual}'")
        
        # Unexpected types
        matched_actuals = {actual for _, actual in partial_matches} | direct_matches
        unexpected = actual_normalized - matched_actuals
        print(f"\n🆕 Unexpected Types: {len(unexpected)}")
        for unexpected_type in sorted(unexpected):
            count = actual_types.get(unexpected_type, 0)
            print(f"  - {unexpected_type}: {count} entities")
        
        # Missing expected types
        all_matches = direct_matches | {expected for expected, _ in partial_matches}
        missing = expected_normalized - all_matches
        print(f"\n❌ Missing Expected Types: {len(missing)}")
        for missing_type in sorted(missing):
            print(f"  - {missing_type}")
            
        # Alignment percentage
        alignment_score = (len(direct_matches) + len(partial_matches) * 0.5) / len(expected_normalized) * 100
        print(f"\n📊 Entity Type Alignment: {alignment_score:.1f}%")
        
    def _analyze_relationship_alignment(self, metadata):
        """Analyze relationship type alignment"""
        relationships = self.data.get('knowledge_data', {}).get('relationships', [])
        
        print("🔗 Relationship Type Alignment")
        print("-" * 30)
        
        # Expected relationship types from metadata
        expected_rel_types = set(metadata.relationship_types.values())
        print(f"Expected Types (from metadata): {', '.join(sorted(expected_rel_types))}")
        
        # Actual relationship types
        actual_rel_types = Counter(rel.get('relation', 'unknown') for rel in relationships)
        print(f"Actual Types Found: {len(actual_rel_types)} unique types")
        
        # Show top actual types
        print(f"\nTop 10 Actual Relationship Types:")
        for rel_type, count in actual_rel_types.most_common(10):
            percentage = (count / len(relationships)) * 100
            print(f"  - {rel_type}: {count} ({percentage:.1f}%)")
        
        # Check alignment with expected types
        expected_normalized = {t.lower().replace('_', ' ') for t in expected_rel_types}
        actual_normalized = {t.lower().replace('_', ' ') for t in actual_rel_types.keys()}
        
        direct_matches = expected_normalized.intersection(actual_normalized)
        
        print(f"\n✅ Direct Matches with Expected: {len(direct_matches)}")
        for match in sorted(direct_matches):
            print(f"  - {match}")
        
        # The system is generating many more specific relationship types than expected
        specificity_score = len(actual_rel_types) / len(expected_rel_types)
        print(f"\n📊 Relationship Specificity: {specificity_score:.1f}x more specific than expected")
        print(f"📊 Relationship Diversity: {len(actual_rel_types)} unique types vs {len(expected_rel_types)} expected")
        
    def _analyze_terminology_alignment(self, patterns):
        """Analyze how well extracted terms align with domain terminology"""
        entities = self.data.get('knowledge_data', {}).get('entities', [])
        relationships = self.data.get('knowledge_data', {}).get('relationships', [])
        
        print("📚 Domain Terminology Alignment")
        print("-" * 32)
        
        # Extract all text from entities and relationships
        entity_texts = [entity.get('text', '').lower() for entity in entities]
        rel_sources = [rel.get('source', '').lower() for rel in relationships]
        rel_targets = [rel.get('target', '').lower() for rel in relationships]
        
        all_extracted_terms = set(entity_texts + rel_sources + rel_targets)
        all_extracted_terms = {term for term in all_extracted_terms if term and len(term) > 2}
        
        # Check alignment with predefined terms
        issue_terms = set(term.lower() for term in patterns.issue_terms)
        action_terms = set(term.lower() for term in patterns.action_terms)
        domain_indicators = set(term.lower() for term in patterns.domain_indicators)
        
        # Find matches
        issue_matches = all_extracted_terms.intersection(issue_terms)
        action_matches = all_extracted_terms.intersection(action_terms)
        domain_matches = all_extracted_terms.intersection(domain_indicators)
        
        print(f"Predefined Issue Terms: {len(issue_terms)}")
        print(f"Found in Extraction: {len(issue_matches)} ({len(issue_matches)/len(issue_terms)*100:.1f}%)")
        print(f"Matches: {', '.join(sorted(issue_matches))}")
        
        print(f"\nPredefined Action Terms: {len(action_terms)}")
        print(f"Found in Extraction: {len(action_matches)} ({len(action_matches)/len(action_terms)*100:.1f}%)")
        print(f"Matches: {', '.join(sorted(action_matches))}")
        
        print(f"\nPredefined Domain Indicators: {len(domain_indicators)}")
        print(f"Found in Extraction: {len(domain_matches)} ({len(domain_matches)/len(domain_indicators)*100:.1f}%)")
        print(f"Matches: {', '.join(sorted(domain_matches))}")
        
        # Check for partial matches (substring matching)
        issue_partials = self._find_partial_matches(all_extracted_terms, issue_terms)
        action_partials = self._find_partial_matches(all_extracted_terms, action_terms)
        domain_partials = self._find_partial_matches(all_extracted_terms, domain_indicators)
        
        if issue_partials:
            print(f"\n🔄 Issue Term Partial Matches: {len(issue_partials)}")
            for extracted, predefined in sorted(issue_partials)[:5]:  # Show first 5
                print(f"  - '{extracted}' ↔ '{predefined}'")
                
        if action_partials:
            print(f"\n🔄 Action Term Partial Matches: {len(action_partials)}")
            for extracted, predefined in sorted(action_partials)[:5]:  # Show first 5
                print(f"  - '{extracted}' ↔ '{predefined}'")
        
        # Overall terminology alignment
        total_predefined = len(issue_terms) + len(action_terms) + len(domain_indicators)
        total_matches = len(issue_matches) + len(action_matches) + len(domain_matches)
        total_partials = len(issue_partials) + len(action_partials) + len(domain_partials)
        
        terminology_score = (total_matches + total_partials * 0.5) / total_predefined * 100
        print(f"\n📊 Overall Terminology Alignment: {terminology_score:.1f}%")
        
    def _find_partial_matches(self, extracted_terms: Set[str], predefined_terms: Set[str]) -> List[tuple]:
        """Find partial matches between extracted and predefined terms"""
        partial_matches = []
        
        for extracted in extracted_terms:
            for predefined in predefined_terms:
                if (extracted in predefined or predefined in extracted) and extracted != predefined:
                    if len(extracted) > 3 and len(predefined) > 3:  # Avoid very short matches
                        partial_matches.append((extracted, predefined))
        
        return partial_matches
        
    def _calculate_alignment_score(self, patterns, prompts, metadata):
        """Calculate overall alignment score"""
        print("🎯 Overall Alignment Score")
        print("-" * 25)
        
        entities = self.data.get('knowledge_data', {}).get('entities', [])
        relationships = self.data.get('knowledge_data', {}).get('relationships', [])
        
        # Entity type alignment (simplified calculation)
        expected_entity_types = set(prompts.extraction_focus.split(', '))
        actual_entity_types = set(entity.get('type', 'unknown') for entity in entities)
        
        # Normalize for comparison
        expected_norm = {t.strip().lower() for t in expected_entity_types}
        actual_norm = {t.lower() for t in actual_entity_types}
        
        entity_overlap = len(expected_norm.intersection(actual_norm))
        entity_alignment = entity_overlap / len(expected_norm) * 100 if expected_norm else 0
        
        # Terminology alignment (simplified)
        all_terms = set()
        for entity in entities:
            all_terms.add(entity.get('text', '').lower())
        for rel in relationships:
            all_terms.add(rel.get('source', '').lower())
            all_terms.add(rel.get('target', '').lower())
        
        predefined_terms = set(patterns.issue_terms + patterns.action_terms + patterns.domain_indicators)
        predefined_terms = {t.lower() for t in predefined_terms}
        
        term_overlap = len(all_terms.intersection(predefined_terms))
        term_alignment = term_overlap / len(predefined_terms) * 100 if predefined_terms else 0
        
        # Quality indicators
        has_context = sum(1 for e in entities if e.get('context', ''))
        context_quality = has_context / len(entities) * 100 if entities else 0
        
        # Overall score (weighted average)
        overall_score = (entity_alignment * 0.4 + term_alignment * 0.4 + context_quality * 0.2)
        
        print(f"Entity Type Alignment: {entity_alignment:.1f}%")
        print(f"Terminology Alignment: {term_alignment:.1f}%")
        print(f"Context Quality: {context_quality:.1f}%")
        print(f"\n🏆 Overall Alignment Score: {overall_score:.1f}%")
        
        # Interpretation
        if overall_score >= 80:
            print("✅ Excellent alignment with domain patterns")
        elif overall_score >= 60:
            print("🟡 Good alignment with some deviations")
        elif overall_score >= 40:
            print("🟠 Moderate alignment - consider pattern refinement")
        else:
            print("🔴 Low alignment - patterns may need significant adjustment")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Analyze alignment between extraction results and domain patterns"
    )
    parser.add_argument(
        "results_file",
        help="Path to knowledge extraction results JSON file"
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = DomainPatternAlignmentAnalyzer(args.results_file)
    analyzer.analyze_alignment()


if __name__ == "__main__":
    main()
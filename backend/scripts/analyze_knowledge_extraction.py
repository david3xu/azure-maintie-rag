#!/usr/bin/env python3
"""
Knowledge Extraction Results Analysis
Comprehensive statistical analysis of extracted entities and relationships
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any
import argparse

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class KnowledgeExtractionAnalyzer:
    """Analyze knowledge extraction results with comprehensive statistics"""
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.data = self._load_results()
        
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
    
    def analyze_all(self):
        """Run comprehensive analysis"""
        print("🔍 Knowledge Extraction Results Analysis")
        print("=" * 50)
        
        # Basic statistics
        self._analyze_basic_stats()
        print()
        
        # Entity analysis
        self._analyze_entities()
        print()
        
        # Relationship analysis
        self._analyze_relationships()
        print()
        
        # Quality metrics
        self._analyze_quality_metrics()
        print()
        
        # Processing statistics
        self._analyze_processing_stats()
        
    def _analyze_basic_stats(self):
        """Basic statistics overview"""
        knowledge_data = self.data.get('knowledge_data', {})
        entities = knowledge_data.get('entities', [])
        relationships = knowledge_data.get('relationships', [])
        
        print("📊 Basic Statistics")
        print("-" * 20)
        print(f"Total Entities: {len(entities):,}")
        print(f"Total Relationships: {len(relationships):,}")
        print(f"Files Processed: {self.data.get('files_processed', 0)}")
        print(f"Domain: {self.data.get('domain', 'unknown')}")
        
        if len(entities) > 0:
            ratio = len(relationships) / len(entities)
            print(f"Relationship-to-Entity Ratio: {ratio:.2f}")
        
        # Calculate items processed (if available)
        total_items = self.data.get('total_items_processed', 'unknown')
        if total_items != 'unknown':
            print(f"Source Texts Processed: {total_items:,}")
            print(f"Entities per Text: {len(entities) / total_items:.2f}")
            print(f"Relationships per Text: {len(relationships) / total_items:.2f}")
    
    def _analyze_entities(self):
        """Detailed entity analysis"""
        entities = self.data.get('knowledge_data', {}).get('entities', [])
        
        print("🏷️  Entity Analysis")
        print("-" * 18)
        
        # Entity types distribution
        entity_types = Counter(entity.get('type', 'unknown') for entity in entities)
        print("Entity Types Distribution:")
        for entity_type, count in entity_types.most_common():
            percentage = (count / len(entities)) * 100
            print(f"  {entity_type}: {count:,} ({percentage:.1f}%)")
        
        # Entity text length analysis
        entity_lengths = [len(entity.get('text', '')) for entity in entities]
        if entity_lengths:
            avg_length = sum(entity_lengths) / len(entity_lengths)
            max_length = max(entity_lengths)
            min_length = min(entity_lengths)
            print(f"\nEntity Text Length:")
            print(f"  Average: {avg_length:.1f} characters")
            print(f"  Range: {min_length} - {max_length} characters")
        
        # Most common entities
        entity_texts = Counter(entity.get('text', '').lower() for entity in entities)
        print(f"\nMost Common Entities:")
        for text, count in entity_texts.most_common(10):
            if count > 1:  # Only show duplicates
                print(f"  '{text}': {count} occurrences")
        
        # Unique entities
        unique_entities = len(set(entity.get('text', '').lower() for entity in entities))
        duplicate_rate = ((len(entities) - unique_entities) / len(entities)) * 100
        print(f"\nEntity Uniqueness:")
        print(f"  Unique entities: {unique_entities:,}")
        print(f"  Duplicate rate: {duplicate_rate:.1f}%")
    
    def _analyze_relationships(self):
        """Detailed relationship analysis"""
        relationships = self.data.get('knowledge_data', {}).get('relationships', [])
        
        print("🔗 Relationship Analysis")
        print("-" * 22)
        
        # Relationship types distribution
        relation_types = Counter(rel.get('relation', 'unknown') for rel in relationships)
        print("Relationship Types Distribution:")
        for rel_type, count in relation_types.most_common():
            percentage = (count / len(relationships)) * 100
            print(f"  {rel_type}: {count:,} ({percentage:.1f}%)")
        
        # Source-target analysis
        sources = Counter(rel.get('source', 'unknown') for rel in relationships)
        targets = Counter(rel.get('target', 'unknown') for rel in relationships)
        
        print(f"\nMost Connected Entities (as source):")
        for source, count in sources.most_common(5):
            print(f"  '{source}': {count} outgoing relationships")
        
        print(f"\nMost Connected Entities (as target):")
        for target, count in targets.most_common(5):
            print(f"  '{target}': {count} incoming relationships")
        
        # Relationship uniqueness
        unique_relations = set()
        for rel in relationships:
            source = rel.get('source', '')
            target = rel.get('target', '')
            relation = rel.get('relation', '')
            unique_relations.add((source.lower(), target.lower(), relation))
        
        duplicate_rel_rate = ((len(relationships) - len(unique_relations)) / len(relationships)) * 100
        print(f"\nRelationship Uniqueness:")
        print(f"  Unique relationships: {len(unique_relations):,}")
        print(f"  Duplicate rate: {duplicate_rel_rate:.1f}%")
    
    def _analyze_quality_metrics(self):
        """Quality and completeness metrics"""
        entities = self.data.get('knowledge_data', {}).get('entities', [])
        relationships = self.data.get('knowledge_data', {}).get('relationships', [])
        
        print("✅ Quality Metrics")
        print("-" * 16)
        
        # Entity completeness
        entities_with_context = sum(1 for e in entities if e.get('context', '').strip())
        entities_with_type = sum(1 for e in entities if e.get('type', '') != 'unknown')
        entities_with_id = sum(1 for e in entities if e.get('entity_id', ''))
        
        print("Entity Completeness:")
        print(f"  With context: {entities_with_context}/{len(entities)} ({entities_with_context/len(entities)*100:.1f}%)")
        print(f"  With type: {entities_with_type}/{len(entities)} ({entities_with_type/len(entities)*100:.1f}%)")
        print(f"  With ID: {entities_with_id}/{len(entities)} ({entities_with_id/len(entities)*100:.1f}%)")
        
        # Relationship completeness
        rels_with_context = sum(1 for r in relationships if r.get('context', '').strip())
        rels_with_type = sum(1 for r in relationships if r.get('relation', '') != 'unknown')
        
        print(f"\nRelationship Completeness:")
        print(f"  With context: {rels_with_context}/{len(relationships)} ({rels_with_context/len(relationships)*100:.1f}%)")
        print(f"  With type: {rels_with_type}/{len(relationships)} ({rels_with_type/len(relationships)*100:.1f}%)")
        
        # Entity coverage in relationships
        entity_texts = set(e.get('text', '').lower() for e in entities)
        rel_entities = set()
        for rel in relationships:
            rel_entities.add(rel.get('source', '').lower())
            rel_entities.add(rel.get('target', '').lower())
        
        connected_entities = entity_texts.intersection(rel_entities)
        isolation_rate = ((len(entity_texts) - len(connected_entities)) / len(entity_texts)) * 100
        
        print(f"\nEntity Connectivity:")
        print(f"  Entities in relationships: {len(connected_entities)}/{len(entity_texts)} ({len(connected_entities)/len(entity_texts)*100:.1f}%)")
        print(f"  Isolated entities: {isolation_rate:.1f}%")
    
    def _analyze_processing_stats(self):
        """Processing performance statistics"""
        print("⏱️  Processing Statistics")
        print("-" * 23)
        
        duration = self.data.get('duration_seconds', 0)
        files_processed = self.data.get('files_processed', 0)
        
        print(f"Total Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Files Processed: {files_processed}")
        
        if duration > 0:
            entities = len(self.data.get('knowledge_data', {}).get('entities', []))
            relationships = len(self.data.get('knowledge_data', {}).get('relationships', []))
            
            print(f"Processing Rate:")
            print(f"  Entities per second: {entities/duration:.2f}")
            print(f"  Relationships per second: {relationships/duration:.2f}")
            print(f"  Items per minute: {(entities + relationships)/duration*60:.1f}")
        
        # Storage information
        saved_to = self.data.get('saved_to', 'Not specified')
        container = self.data.get('azure_container', 'Not specified')
        
        print(f"\nStorage Information:")
        print(f"  Azure Container: {container}")
        print(f"  Azure Blob: {saved_to}")
        print(f"  Local File: {self.results_file}")
    
    def generate_summary_report(self, output_file: str = None):
        """Generate a summary report"""
        entities = self.data.get('knowledge_data', {}).get('entities', [])
        relationships = self.data.get('knowledge_data', {}).get('relationships', [])
        
        # Entity types summary
        entity_types = Counter(entity.get('type', 'unknown') for entity in entities)
        relation_types = Counter(rel.get('relation', 'unknown') for rel in relationships)
        
        summary = {
            "extraction_summary": {
                "total_entities": len(entities),
                "total_relationships": len(relationships),
                "files_processed": self.data.get('files_processed', 0),
                "domain": self.data.get('domain', 'unknown'),
                "processing_time_minutes": round(self.data.get('duration_seconds', 0) / 60, 1)
            },
            "entity_types": dict(entity_types.most_common()),
            "relationship_types": dict(relation_types.most_common()),
            "quality_metrics": {
                "unique_entities": len(set(e.get('text', '').lower() for e in entities)),
                "entities_with_context": sum(1 for e in entities if e.get('context', '').strip()),
                "relationships_with_context": sum(1 for r in relationships if r.get('context', '').strip())
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"📄 Summary report saved to: {output_file}")
        
        return summary


def main():
    """Main entry point for knowledge extraction analysis"""
    parser = argparse.ArgumentParser(
        description="Analyze knowledge extraction results with comprehensive statistics"
    )
    parser.add_argument(
        "results_file",
        help="Path to knowledge extraction results JSON file"
    )
    parser.add_argument(
        "--summary",
        help="Generate summary report to specified file"
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = KnowledgeExtractionAnalyzer(args.results_file)
    analyzer.analyze_all()
    
    if args.summary:
        print("\n" + "="*50)
        analyzer.generate_summary_report(args.summary)


if __name__ == "__main__":
    main()
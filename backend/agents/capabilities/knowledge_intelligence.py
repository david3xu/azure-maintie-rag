"""
Knowledge Service
Handles all knowledge extraction and processing operations
Consolidates: extraction workflows, text processing, entity/relationship management
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
from pathlib import Path

from infra.azure_openai import UnifiedAzureOpenAIClient
from infra.azure_storage import UnifiedStorageClient

logger = logging.getLogger(__name__)


class KnowledgeService:
    """High-level service for knowledge extraction and processing"""
    
    def __init__(self):
        self.openai_client = UnifiedAzureOpenAIClient()
        self.storage_client = UnifiedStorageClient()
        
    # === MAIN EXTRACTION WORKFLOWS ===
    
    async def extract_from_file(self, file_path: str, domain: str = "maintenance") -> Dict[str, Any]:
        """Extract knowledge from a text file"""
        try:
            # Load text data
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Process text into individual entries
            texts = self._parse_text_file(content)
            
            # Extract knowledge
            result = await self.openai_client.extract_knowledge(texts, domain)
            
            if result['success']:
                # Save results to storage
                timestamp = self._get_timestamp()
                blob_name = f"extraction_{domain}_{timestamp}.json"
                
                save_result = await self.storage_client.save_json(
                    result['data'], 
                    blob_name,
                    container="extractions"
                )
                
                result['data']['saved_to'] = blob_name
                
            return result
            
        except Exception as e:
            logger.error(f"Extraction from file failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'extract_from_file'
            }
    
    async def extract_from_texts(self, texts: List[str], domain: str = "maintenance") -> Dict[str, Any]:
        """Extract knowledge from list of texts"""
        try:
            # Clean and filter texts
            cleaned_texts = [self._clean_text(text) for text in texts if text.strip()]
            
            # Extract knowledge using Azure OpenAI
            result = await self.openai_client.extract_knowledge(cleaned_texts, domain)
            
            return result
            
        except Exception as e:
            logger.error(f"Extraction from texts failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'extract_from_texts'
            }
    
    # === DATA PROCESSING ===
    
    def process_extraction_results(self, entities: List[Dict], relationships: List[Dict]) -> Dict[str, Any]:
        """Process and validate extraction results"""
        try:
            # Validate entities
            valid_entities = self._validate_entities(entities)
            
            # Validate relationships
            valid_relationships = self._validate_relationships(relationships, valid_entities)
            
            # Generate statistics
            stats = self._generate_extraction_stats(valid_entities, valid_relationships)
            
            return {
                'success': True,
                'entities': valid_entities,
                'relationships': valid_relationships,
                'statistics': stats
            }
            
        except Exception as e:
            logger.error(f"Processing extraction results failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'process_extraction_results'
            }
    
    def merge_extractions(self, extraction_files: List[str]) -> Dict[str, Any]:
        """Merge multiple extraction results"""
        try:
            all_entities = []
            all_relationships = []
            
            for file_path in extraction_files:
                with open(file_path, 'r') as f:
                    import json
                    data = json.load(f)
                    
                    all_entities.extend(data.get('entities', []))
                    all_relationships.extend(data.get('relationships', []))
            
            # Deduplicate and process
            merged_result = self.process_extraction_results(all_entities, all_relationships)
            
            return merged_result
            
        except Exception as e:
            logger.error(f"Merging extractions failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'operation': 'merge_extractions'
            }
    
    # === TEXT PROCESSING UTILITIES ===
    
    def _parse_text_file(self, content: str) -> List[str]:
        """Parse text file into individual maintenance entries"""
        # Parse structured content with <id> markers (MaintIE format)
        if '<id>' in content:
            # Split by <id> markers and extract maintenance texts
            maintenance_items = content.split('<id>')
            texts = []
            
            for item in maintenance_items[1:]:  # Skip first empty split
                item = item.strip()
                if item and len(item) > 10:  # Filter out very short items
                    texts.append(item)
                    
            logger.info(f"📄 Parsed {len(texts)} maintenance items from structured data")
            return texts
        else:
            # Fallback to line-based parsing
            lines = content.split('\n')  # Fixed: was using \\n
            texts = []
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and len(line) > 20:
                    texts.append(line)
            
            logger.info(f"📄 Parsed {len(texts)} lines from unstructured data")
            return texts
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        import re
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep maintenance-relevant punctuation
        text = re.sub(r'[^\w\s\.,!?;:()-]', '', text)
        return text
    
    # === VALIDATION ===
    
    def _validate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Validate and clean entities"""
        valid_entities = []
        seen_texts = set()
        
        for entity in entities:
            text = entity.get('text', '').strip().lower()
            entity_type = entity.get('type', 'unknown')
            
            # Skip empty or duplicate entities
            if not text or text in seen_texts:
                continue
                
            # Validate entity type
            if entity_type not in ['equipment', 'component', 'issue', 'action', 'location', 'person', 'unknown']:
                entity_type = 'unknown'
            
            seen_texts.add(text)
            valid_entities.append({
                'entity_id': f"entity_{len(valid_entities)}",
                'text': entity.get('text', '').strip(),
                'entity_type': entity_type,
                'context': entity.get('context', '')[:200]  # Limit context length
            })
        
        return valid_entities
    
    def _validate_relationships(self, relationships: List[Dict], entities: List[Dict]) -> List[Dict]:
        """Validate relationships against entities"""
        entity_texts = {entity['text'].lower(): entity['entity_id'] for entity in entities}
        valid_relationships = []
        seen_relations = set()
        
        for rel in relationships:
            source_text = rel.get('source', '').strip().lower()
            target_text = rel.get('target', '').strip().lower()
            relation_type = rel.get('relation', 'related')
            
            # Check if both entities exist
            if source_text in entity_texts and target_text in entity_texts:
                source_id = entity_texts[source_text]
                target_id = entity_texts[target_text]
                
                # Avoid duplicate relationships
                rel_key = (source_id, target_id, relation_type)
                if rel_key not in seen_relations:
                    seen_relations.add(rel_key)
                    valid_relationships.append({
                        'source_entity_id': source_id,
                        'target_entity_id': target_id,
                        'relation_type': relation_type,
                        'confidence': rel.get('confidence', 1.0),
                        'context': rel.get('context', '')[:200]
                    })
        
        return valid_relationships
    
    # === STATISTICS ===
    
    def _generate_extraction_stats(self, entities: List[Dict], relationships: List[Dict]) -> Dict[str, Any]:
        """Generate extraction statistics"""
        # Entity type distribution
        entity_types = {}
        for entity in entities:
            entity_type = entity['entity_type']
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        # Relationship type distribution
        relation_types = {}
        for rel in relationships:
            rel_type = rel['relation_type']
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
        
        return {
            'total_entities': len(entities),
            'total_relationships': len(relationships),
            'entity_types': entity_types,
            'relation_types': relation_types,
            'connectivity_ratio': len(relationships) / len(entities) if entities else 0
        }
    
    def _get_timestamp(self) -> str:
        """Get timestamp for file naming"""
        from datetime import datetime
        return datetime.now().strftime('%Y%m%d_%H%M%S')
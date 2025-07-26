"""
Dual Storage Knowledge Extractor
Outputs extraction results locally AND stores in Azure services
Provides comprehensive comparison with raw text data
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib

from .improved_extraction_client import ImprovedKnowledgeExtractor
from ..azure_storage.real_azure_services import AzureBlobService, AzureSearchService, AzureCosmosService
from config.settings import settings

logger = logging.getLogger(__name__)


class DualStorageExtractor:
    """
    Knowledge extractor that stores results both locally and in Azure services
    Provides detailed comparison outputs with raw text data
    """
    
    def __init__(self, domain_name: str = "maintenance"):
        self.domain_name = domain_name
        
        # Initialize improved extractor
        self.extractor = ImprovedKnowledgeExtractor(domain_name)
        
        # Local storage paths
        self.local_output_dir = Path(settings.BASE_DIR) / "data" / "extraction_outputs"
        self.local_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.raw_data_dir = self.local_output_dir / "raw_texts"
        self.extracted_data_dir = self.local_output_dir / "extracted_knowledge"
        self.comparison_dir = self.local_output_dir / "comparisons"
        
        for dir_path in [self.raw_data_dir, self.extracted_data_dir, self.comparison_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Azure services
        try:
            self.blob_service = AzureBlobService()
            self.search_service = AzureSearchService() 
            self.cosmos_service = AzureCosmosService()
            self.azure_available = True
            logger.info("Azure services initialized successfully")
        except Exception as e:
            logger.warning(f"Azure services not available: {e}")
            self.azure_available = False
        
        logger.info(f"DualStorageExtractor initialized - Azure available: {self.azure_available}")
    
    def extract_and_store_all(self, texts: List[str], batch_name: str = None) -> Dict[str, Any]:
        """
        Extract knowledge and store in both local files and Azure services
        """
        if not batch_name:
            batch_name = f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting dual storage extraction for {len(texts)} texts - Batch: {batch_name}")
        
        # Create batch directory
        batch_dir = self.local_output_dir / batch_name
        batch_dir.mkdir(exist_ok=True)
        
        results = {
            "batch_name": batch_name,
            "timestamp": datetime.now().isoformat(),
            "domain": self.domain_name,
            "total_texts": len(texts),
            "local_storage": {
                "raw_texts_file": None,
                "extracted_knowledge_file": None,
                "comparison_file": None
            },
            "azure_storage": {
                "blob_container": settings.azure_blob_container if self.azure_available else "not_available",
                "search_index": "knowledge-extraction-index",
                "cosmos_container": "extracted-knowledge",
                "upload_status": "pending"
            },
            "extraction_results": [],
            "storage_summary": {}
        }
        
        # Process each text
        all_extractions = []
        for i, raw_text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            
            # Generate unique ID for this text
            text_id = hashlib.md5(raw_text.encode()).hexdigest()[:12]
            
            # Extract knowledge
            extraction = self.extractor._extract_from_single_text(raw_text)
            
            # Add metadata
            extraction_record = {
                "text_id": text_id,
                "sequence_number": i,
                "raw_text": raw_text,
                "extraction_timestamp": datetime.now().isoformat(),
                "batch_name": batch_name,
                "domain": self.domain_name,
                **extraction
            }
            
            all_extractions.append(extraction_record)
            results["extraction_results"].append(extraction_record)
            
            # Store individual text locally
            self._store_individual_text_locally(extraction_record, batch_dir)
            
            # Print real-time comparison
            self._print_real_time_comparison(i, raw_text, extraction)
        
        # Store batch results locally
        local_files = self._store_batch_locally(all_extractions, batch_name)
        results["local_storage"].update(local_files)
        
        # Store in Azure services
        if self.azure_available:
            azure_status = self._store_batch_in_azure(all_extractions, batch_name)
            results["azure_storage"]["upload_status"] = azure_status
        
        # Generate storage summary
        results["storage_summary"] = self._generate_storage_summary(all_extractions, results)
        
        # Save master results file
        master_file = self.local_output_dir / f"{batch_name}_master_results.json"
        with open(master_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Dual storage extraction completed - Results in: {master_file}")
        
        return results
    
    def _store_individual_text_locally(self, extraction_record: Dict[str, Any], batch_dir: Path):
        """Store individual text extraction locally"""
        text_id = extraction_record["text_id"]
        
        # Raw text file
        raw_file = batch_dir / f"{text_id}_raw.txt"
        with open(raw_file, 'w') as f:
            f.write(f"Text ID: {text_id}\n")
            f.write(f"Timestamp: {extraction_record['extraction_timestamp']}\n")
            f.write(f"Batch: {extraction_record['batch_name']}\n")
            f.write(f"Domain: {extraction_record['domain']}\n")
            f.write("-" * 50 + "\n")
            f.write(extraction_record["raw_text"])
        
        # Extracted knowledge file
        extraction_file = batch_dir / f"{text_id}_extracted.json"
        extraction_data = {
            "text_id": text_id,
            "raw_text": extraction_record["raw_text"],
            "entities": extraction_record["entities"],
            "relations": extraction_record["relations"],
            "metadata": {
                "extraction_timestamp": extraction_record["extraction_timestamp"],
                "batch_name": extraction_record["batch_name"],
                "domain": extraction_record["domain"]
            }
        }
        
        with open(extraction_file, 'w') as f:
            json.dump(extraction_data, f, indent=2)
        
        # Comparison file
        comparison_file = batch_dir / f"{text_id}_comparison.md"
        self._create_comparison_markdown(extraction_record, comparison_file)
    
    def _store_batch_locally(self, extractions: List[Dict[str, Any]], batch_name: str) -> Dict[str, str]:
        """Store batch results in local files"""
        
        # All raw texts
        raw_texts_file = self.raw_data_dir / f"{batch_name}_raw_texts.jsonl"
        with open(raw_texts_file, 'w') as f:
            for extraction in extractions:
                raw_record = {
                    "text_id": extraction["text_id"],
                    "raw_text": extraction["raw_text"],
                    "timestamp": extraction["extraction_timestamp"],
                    "batch_name": batch_name
                }
                f.write(json.dumps(raw_record) + "\n")
        
        # All extracted knowledge
        extracted_file = self.extracted_data_dir / f"{batch_name}_extracted_knowledge.jsonl"
        with open(extracted_file, 'w') as f:
            for extraction in extractions:
                knowledge_record = {
                    "text_id": extraction["text_id"],
                    "entities": extraction["entities"],
                    "relations": extraction["relations"],
                    "extraction_metadata": {
                        "timestamp": extraction["extraction_timestamp"],
                        "batch_name": batch_name,
                        "domain": extraction["domain"]
                    }
                }
                f.write(json.dumps(knowledge_record) + "\n")
        
        # Comprehensive comparison
        comparison_file = self.comparison_dir / f"{batch_name}_complete_comparison.json"
        comparison_data = {
            "batch_info": {
                "batch_name": batch_name,
                "total_texts": len(extractions),
                "domain": self.domain_name,
                "timestamp": datetime.now().isoformat()
            },
            "detailed_comparisons": []
        }
        
        for extraction in extractions:
            quality = self.extractor._assess_extraction_quality(extraction["raw_text"], extraction)
            issues = self.extractor._identify_issues(extraction["raw_text"], extraction)
            
            comparison_data["detailed_comparisons"].append({
                "text_id": extraction["text_id"],
                "raw_text": extraction["raw_text"],
                "entities_extracted": extraction["entities"],
                "relations_extracted": extraction["relations"],
                "quality_metrics": quality,
                "issues_identified": issues
            })
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        return {
            "raw_texts_file": str(raw_texts_file),
            "extracted_knowledge_file": str(extracted_file),
            "comparison_file": str(comparison_file)
        }
    
    def _store_batch_in_azure(self, extractions: List[Dict[str, Any]], batch_name: str) -> str:
        """Store batch results in Azure services"""
        try:
            upload_results = {
                "blob_storage": False,
                "search_service": False,
                "cosmos_db": False
            }
            
            # Upload to Azure Blob Storage
            try:
                blob_data = {
                    "batch_name": batch_name,
                    "extractions": extractions,
                    "timestamp": datetime.now().isoformat()
                }
                
                blob_name = f"knowledge-extractions/{batch_name}.json"
                self.blob_service.upload_json(blob_data, blob_name)
                upload_results["blob_storage"] = True
                logger.info(f"Uploaded to Azure Blob: {blob_name}")
                
            except Exception as e:
                logger.error(f"Azure Blob upload failed: {e}")
            
            # Index in Azure Cognitive Search
            try:
                search_documents = []
                for extraction in extractions:
                    # Create search document for each entity
                    for entity in extraction["entities"]:
                        search_doc = {
                            "id": f"{extraction['text_id']}_{entity['entity_id']}",
                            "text_id": extraction["text_id"],
                            "batch_name": batch_name,
                            "raw_text": extraction["raw_text"],
                            "entity_text": entity["text"],
                            "entity_type": entity.get("entity_type", "unknown"),
                            "context": entity.get("context", ""),
                            "domain": self.domain_name,
                            "timestamp": extraction["extraction_timestamp"]
                        }
                        search_documents.append(search_doc)
                
                if search_documents:
                    self.search_service.upload_documents(search_documents)
                    upload_results["search_service"] = True
                    logger.info(f"Indexed {len(search_documents)} documents in Azure Search")
                
            except Exception as e:
                logger.error(f"Azure Search indexing failed: {e}")
            
            # Store in Azure Cosmos DB
            try:
                for extraction in extractions:
                    cosmos_doc = {
                        "id": extraction["text_id"],
                        "partitionKey": batch_name,
                        "raw_text": extraction["raw_text"],
                        "entities": extraction["entities"],
                        "relations": extraction["relations"],
                        "batch_name": batch_name,
                        "domain": self.domain_name,
                        "extraction_timestamp": extraction["extraction_timestamp"]
                    }
                    
                    self.cosmos_service.upsert_document(cosmos_doc)
                
                upload_results["cosmos_db"] = True
                logger.info(f"Stored {len(extractions)} documents in Cosmos DB")
                
            except Exception as e:
                logger.error(f"Cosmos DB storage failed: {e}")
            
            # Determine overall status
            successful_uploads = sum(upload_results.values())
            if successful_uploads == 3:
                return "success_all"
            elif successful_uploads > 0:
                return f"partial_success_{successful_uploads}/3"
            else:
                return "failed_all"
                
        except Exception as e:
            logger.error(f"Azure storage failed: {e}")
            return "failed_error"
    
    def _create_comparison_markdown(self, extraction_record: Dict[str, Any], output_file: Path):
        """Create detailed markdown comparison"""
        content = f"""# Knowledge Extraction Comparison

## Text Information
- **Text ID:** {extraction_record['text_id']}
- **Batch:** {extraction_record['batch_name']}
- **Domain:** {extraction_record['domain']}
- **Timestamp:** {extraction_record['extraction_timestamp']}

## Raw Text
```
{extraction_record['raw_text']}
```

## Extracted Entities
"""
        
        for entity in extraction_record['entities']:
            content += f"""
### {entity.get('text', 'Unknown')}
- **Type:** {entity.get('entity_type', 'unknown')}
- **Context:** "{entity.get('context', 'N/A')}"
- **ID:** {entity.get('entity_id', 'N/A')}
"""
        
        content += "\n## Extracted Relations\n"
        
        for relation in extraction_record['relations']:
            source_id = relation.get('source_entity_id', '')
            target_id = relation.get('target_entity_id', '')
            
            # Find entity texts
            source_text = next((e['text'] for e in extraction_record['entities'] if e['entity_id'] == source_id), source_id)
            target_text = next((e['text'] for e in extraction_record['entities'] if e['entity_id'] == target_id), target_id)
            
            content += f"""
### {source_text} → {target_text}
- **Relation:** {relation.get('relation_type', 'unknown')}
- **Context:** "{relation.get('context', 'N/A')}"
- **Confidence:** {relation.get('confidence', 'N/A')}
"""
        
        # Add quality assessment
        quality = self.extractor._assess_extraction_quality(extraction_record['raw_text'], extraction_record)
        content += f"""
## Quality Assessment
- **Entities Extracted:** {quality['entities_extracted']}
- **Relations Extracted:** {quality['relations_extracted']}
- **Context Preservation:** {quality['context_preservation_score']}
- **Entity Coverage:** {quality['entity_coverage_score']}
- **Connectivity:** {quality['connectivity_score']}
- **Overall Quality:** {quality['overall_quality']}
"""
        
        with open(output_file, 'w') as f:
            f.write(content)
    
    def _print_real_time_comparison(self, index: int, raw_text: str, extraction: Dict[str, Any]):
        """Print real-time comparison to console"""
        print(f"\n{'='*80}")
        print(f"TEXT {index + 1}: DUAL STORAGE EXTRACTION")
        print(f"{'='*80}")
        
        print(f"\n📝 RAW TEXT:")
        print(f'   "{raw_text}"')
        
        print(f"\n🔍 EXTRACTED ENTITIES:")
        for entity in extraction.get('entities', []):
            print(f"   • {entity.get('text', 'N/A')} [{entity.get('entity_type', 'unknown')}]")
        
        print(f"\n🔗 EXTRACTED RELATIONS:")
        for relation in extraction.get('relations', []):
            source_id = relation.get('source_entity_id', '')
            target_id = relation.get('target_entity_id', '')
            
            # Find entity texts
            source_text = next((e['text'] for e in extraction['entities'] if e['entity_id'] == source_id), source_id)
            target_text = next((e['text'] for e in extraction['entities'] if e['entity_id'] == target_id), target_id)
            
            print(f"   • {source_text} --[{relation.get('relation_type', 'unknown')}]--> {target_text}")
        
        print(f"\n💾 STORAGE:")
        print(f"   • Local: Individual files created")
        print(f"   • Azure: {'Queued for upload' if self.azure_available else 'Not available'}")
    
    def _generate_storage_summary(self, extractions: List[Dict[str, Any]], results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive storage summary"""
        total_entities = sum(len(e['entities']) for e in extractions)
        total_relations = sum(len(e['relations']) for e in extractions)
        
        return {
            "extraction_stats": {
                "total_texts": len(extractions),
                "total_entities": total_entities,
                "total_relations": total_relations,
                "avg_entities_per_text": round(total_entities / len(extractions), 2),
                "avg_relations_per_text": round(total_relations / len(extractions), 2)
            },
            "storage_locations": {
                "local_files": len(list(self.local_output_dir.rglob("*"))) if self.local_output_dir.exists() else 0,
                "azure_blob": "uploaded" if self.azure_available and results["azure_storage"]["upload_status"].startswith("success") else "not_uploaded",
                "azure_search": "indexed" if self.azure_available else "not_available",
                "azure_cosmos": "stored" if self.azure_available else "not_available"
            },
            "file_paths": {
                "local_output_directory": str(self.local_output_dir),
                "batch_directory": str(self.local_output_dir / results["batch_name"]),
                "master_results": str(self.local_output_dir / f"{results['batch_name']}_master_results.json")
            }
        }


if __name__ == "__main__":
    # Test dual storage extraction
    extractor = DualStorageExtractor("maintenance")
    
    sample_texts = [
        "air conditioner thermostat not working",
        "air receiver safety valves to be replaced",
        "analyse failed driveline component"
    ]
    
    results = extractor.extract_and_store_all(sample_texts, "test_dual_storage")
    print(f"\n🎯 DUAL STORAGE COMPLETE")
    print(f"Local files: {results['storage_summary']['file_paths']['local_output_directory']}")
    print(f"Azure status: {results['azure_storage']['upload_status']}")
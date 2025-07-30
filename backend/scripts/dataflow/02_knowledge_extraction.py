#!/usr/bin/env python3
"""
Knowledge Extraction - Stage 2 of README Data Flow
Raw Text Data → Knowledge Extraction (Azure OpenAI)

This script implements the second stage of the README architecture:
- Uses existing KnowledgeService to extract entities and relationships
- Processes data from Azure Storage containers
- Extracts knowledge using Azure OpenAI LLM
- Prepares structured knowledge for next pipeline stages
"""

import sys
import asyncio
import argparse
import json
from pathlib import Path
from typing import Dict, Any
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.knowledge_service import KnowledgeService
from services.infrastructure_service import InfrastructureService

logger = logging.getLogger(__name__)

class KnowledgeExtractionStage:
    """Stage 2: Raw Text Data → Knowledge Extraction (using existing KnowledgeService)"""
    
    def __init__(self):
        self.infrastructure = InfrastructureService()
        self.knowledge_service = KnowledgeService()
    
    async def _save_progress(self, storage_client, progress_file: str, progress_data: Dict[str, Any]):
        """Save incremental progress to local file"""
        try:
            # Ensure directory exists
            Path(progress_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Save locally with safe JSON handling
            from core.utilities.file_utils import FileUtils
            with open(progress_file, 'w') as f:
                f.write(FileUtils.safe_json_dumps(progress_data, indent=2))
                
        except Exception as e:
            print(f"⚠️ Warning: Could not save progress: {e}")
        
    async def execute(self, container_name: str = None, domain: str = "maintenance") -> Dict[str, Any]:
        """
        Execute knowledge extraction using Azure Storage data
        
        Args:
            container_name: Azure Storage container name (optional, auto-determined if not provided)
            domain: Domain for extraction strategy
            
        Returns:
            Dict with extraction results and structured knowledge
        """
        print("🧠 Stage 2: Knowledge Extraction - Raw Text → Azure OpenAI")
        print("=" * 58)
        
        start_time = asyncio.get_event_loop().time()
        
        # Determine container name
        if not container_name:
            from config.domain_patterns import DomainPatternManager
            container_name = DomainPatternManager.get_container_name(domain, "data")
        
        results = {
            "stage": "02_knowledge_extraction",
            "azure_container": container_name,
            "domain": domain,
            "files_processed": 0,
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "knowledge_data": {
                "entities": [],
                "relationships": []
            },
            "success": False
        }
        
        try:
            # Get Azure Storage client
            storage_client = self.infrastructure.storage_client
            if not storage_client:
                raise RuntimeError("❌ Azure Storage client not initialized")
            
            # List files in Azure container
            print(f"📦 Retrieving files from Azure container: {container_name}")
            blobs_result = await storage_client.list_blobs(container_name)
            
            if not blobs_result.get('success'):
                raise RuntimeError(f"Failed to list blobs: {blobs_result.get('error')}")
            
            blobs = blobs_result.get('data', {}).get('blobs', [])
            print(f"📄 Found {len(blobs)} files in Azure Storage")
            
            all_entities = []
            all_relationships = []
            files_processed = 0
            
            # Check for existing progress file to resume from
            progress_file = "data/outputs/step02_progress.json"
            resume_from = 0
            if Path(progress_file).exists():
                try:
                    with open(progress_file, 'r') as f:
                        progress_data = json.load(f)
                        if progress_data.get('container') == container_name and progress_data.get('domain') == domain:
                            resume_from = progress_data.get('last_processed_index', 0)
                            all_entities = progress_data.get('entities', [])
                            all_relationships = progress_data.get('relationships', [])
                            print(f"🔄 Resuming from item {resume_from + 1} (found {len(all_entities)} existing entities)")
                except Exception as e:
                    print(f"⚠️ Could not load progress file: {e}. Starting fresh.")
                    resume_from = 0
            
            # Process each file from Azure Storage
            for blob in blobs:
                blob_name = blob.get('name') if isinstance(blob, dict) else blob
                print(f"🔍 Processing: {blob_name}")
                
                # Download file content from Azure
                download_result = await storage_client.download_data(blob_name, container_name)
                if not download_result.get('success'):
                    print(f"❌ Failed to download {blob_name}: {download_result.get('error')}")
                    continue
                
                data_info = download_result.get('data', {})
                file_content = data_info.get('data', '')
                if not file_content:
                    print(f"⚠️  Empty content in {blob_name}")
                    continue
                
                # Parse the file content into individual maintenance items
                parsed_texts = self.knowledge_service._parse_text_file(file_content)
                print(f"📝 Parsed {len(parsed_texts)} maintenance items from {blob_name}")
                print(f"⏱️  Estimated processing time: {(len(parsed_texts) - resume_from) * 2 / 60:.1f} minutes")
                print(f"🔄 Starting extraction from item {resume_from + 1} to {len(parsed_texts)}...")
                
                # Process items individually with incremental saving
                import time
                start_extract_time = time.time()
                
                for idx in range(resume_from, len(parsed_texts)):
                    text = parsed_texts[idx]
                    
                    # Extract from single text
                    single_result = await self.knowledge_service.extract_from_texts([text], domain)
                    
                    if single_result.get('success'):
                        single_data = single_result.get('data', {})
                        entities = single_data.get('entities', [])
                        relationships = single_data.get('relationships', [])
                        
                        all_entities.extend(entities)
                        all_relationships.extend(relationships)
                        
                        # Progress display
                        if (idx + 1) % 5 == 0 or idx == 0:
                            elapsed = time.time() - start_extract_time
                            remaining_items = len(parsed_texts) - idx - 1  
                            remaining_time = (elapsed / (idx - resume_from + 1)) * remaining_items if idx > resume_from else 0
                            print(f"    📊 Progress: {idx + 1}/{len(parsed_texts)} | Entities: {len(all_entities)} | Relationships: {len(all_relationships)} | ETA: {remaining_time/60:.1f}min")
                        
                        # Incremental save every 20 items
                        if (idx + 1) % 20 == 0:
                            await self._save_progress(storage_client, progress_file, {
                                'container': container_name,
                                'domain': domain,
                                'last_processed_index': idx,
                                'entities': all_entities,
                                'relationships': all_relationships,
                                'timestamp': time.time()
                            })
                            print(f"    💾 Progress saved at item {idx + 1}")
                    else:
                        print(f"    ❌ Failed to extract from item {idx + 1}: {single_result.get('error')}")
                
                end_extract_time = time.time()
                extract_duration = end_extract_time - start_extract_time
                
                files_processed += 1
                print(f"✅ File complete: {len(all_entities)} total entities, {len(all_relationships)} total relationships")
                print(f"⏱️  Processing time: {extract_duration:.1f} seconds ({extract_duration/60:.1f} minutes)")
                
                # Clean up progress file on successful completion
                if Path(progress_file).exists():
                    Path(progress_file).unlink()
                    print(f"🧹 Progress file cleaned up")
            
            # Combine all extracted knowledge
            entities = all_entities
            relationships = all_relationships
            
            # Save extracted knowledge to Azure Storage
            if entities or relationships:
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                knowledge_blob_name = f"knowledge_extraction_{domain}_{timestamp}.json"
                
                knowledge_data = {
                    "entities": entities,
                    "relationships": relationships,
                    "source_container": container_name,
                    "files_processed": files_processed,
                    "extraction_timestamp": timestamp,
                    "domain": domain
                }
                
                save_result = await storage_client.save_json(
                    knowledge_data,
                    knowledge_blob_name,
                    container="extractions"
                )
                
                if save_result.get('success'):
                    print(f"💾 Knowledge data saved to: {knowledge_blob_name}")
                    results["saved_to"] = knowledge_blob_name
                else:
                    print(f"❌ Failed to save knowledge data: {save_result.get('error')}")
            
            # Update results
            results.update({
                "files_processed": files_processed,
                "entities_extracted": len(entities),
                "relationships_extracted": len(relationships),
                "knowledge_data": {
                    "entities": entities,
                    "relationships": relationships
                },
                "azure_blobs_processed": [blob.get('name') if isinstance(blob, dict) else blob for blob in blobs]
            })
            
            # Success
            duration = asyncio.get_event_loop().time() - start_time
            results["duration_seconds"] = round(duration, 2)
            results["success"] = True
            
            # Storage info from KnowledgeService
            if extraction_result.get('data', {}).get('saved_to'):
                results['saved_to'] = extraction_result['data']['saved_to']
            
            print(f"✅ Stage 2 Complete:")
            print(f"   📄 Files processed: {results['files_processed']}")
            print(f"   🏷️  Entities extracted: {results['entities_extracted']}")
            print(f"   🔗 Relationships extracted: {results['relationships_extracted']}")
            print(f"   ⏱️  Duration: {results['duration_seconds']}s")
            if results.get('saved_to'):
                print(f"   💾 Knowledge saved to: {results['saved_to']}")
            
            return results
            
        except Exception as e:
            results["error"] = str(e)
            results["duration_seconds"] = round(asyncio.get_event_loop().time() - start_time, 2)
            print(f"❌ Stage 2 Failed: {e}")
            logger.error(f"Knowledge extraction failed: {e}", exc_info=True)
            return results


async def main():
    """Main entry point for knowledge extraction stage"""
    parser = argparse.ArgumentParser(
        description="Stage 2: Knowledge Extraction - Raw Text → Azure OpenAI"
    )
    parser.add_argument(
        "--container", 
        help="Azure Storage container name (auto-determined if not provided)"
    )
    parser.add_argument(
        "--domain", 
        default="maintenance",
        help="Domain for extraction strategy"
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Execute stage
    stage = KnowledgeExtractionStage()
    results = await stage.execute(
        container_name=args.container,
        domain=args.domain
    )
    
    # Save results if requested
    if args.output and results.get("success"):
        from core.utilities.file_utils import FileUtils
        with open(args.output, 'w') as f:
            f.write(FileUtils.safe_json_dumps(results, indent=2))
        print(f"📄 Results saved to: {args.output}")
    
    # Return appropriate exit code
    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
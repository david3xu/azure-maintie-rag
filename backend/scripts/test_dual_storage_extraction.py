#!/usr/bin/env python3
"""
Test Dual Storage Knowledge Extraction
Shows extraction results stored locally AND in mock Azure services
Provides comprehensive comparison with raw text data
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Mock the Azure services first
import core.azure_storage.mock_azure_services as mock_services

# Monkey patch the imports in dual_storage_extractor
sys.modules['core.azure_storage.azure_blob_service'] = type('MockModule', (), {
    'AzureBlobService': mock_services.MockAzureBlobService
})
sys.modules['core.azure_search.azure_search_service'] = type('MockModule', (), {
    'AzureSearchService': mock_services.MockAzureSearchService  
})
sys.modules['core.azure_cosmos.cosmos_service'] = type('MockModule', (), {
    'AzureCosmosService': mock_services.MockAzureCosmosService
})

from core.azure_openai.dual_storage_extractor import DualStorageExtractor


def load_sample_texts(limit: int = 15) -> list[str]:
    """Load sample maintenance texts"""
    
    data_file = Path(__file__).parent.parent / "data" / "raw" / "maintenance_all_texts.md"
    
    if not data_file.exists():
        # Fallback sample texts
        return [
            "air conditioner thermostat not working",
            "air receiver safety valves to be replaced", 
            "analyse failed driveline component",
            "auxiliary Cat engine lube service",
            "axle temperature sensor fault",
            "back rest unserviceable handle broken",
            "backhoe windscreen to be fixed",
            "backlight on dash unserviceable",
            "auto-greaser control unit",
            "alarm on VIMS doesn't work",
            "alternator overcharge fault",
            "air horn not working compressor awaiting",
            "air horn working intermittently",
            "air leak near side of door",
            "air leaking from line outside"
        ]
    
    texts = []
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('<id>') and len(line) > 5:
                text = line[4:].strip()
                if text and len(text) > 10:
                    texts.append(text)
                    if len(texts) >= limit:
                        break
    
    return texts


def display_storage_results(results: dict):
    """Display comprehensive storage results"""
    
    print(f"\n{'='*80}")
    print("📁 STORAGE RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Basic info
    print(f"\n📊 Batch Information:")
    print(f"   • Batch Name: {results['batch_name']}")
    print(f"   • Domain: {results['domain']}")
    print(f"   • Total Texts: {results['total_texts']}")
    print(f"   • Timestamp: {results['timestamp']}")
    
    # Extraction stats
    summary = results.get('storage_summary', {})
    stats = summary.get('extraction_stats', {})
    print(f"\n📈 Extraction Statistics:")
    print(f"   • Total Entities: {stats.get('total_entities', 0)}")
    print(f"   • Total Relations: {stats.get('total_relations', 0)}")
    print(f"   • Avg Entities/Text: {stats.get('avg_entities_per_text', 0)}")
    print(f"   • Avg Relations/Text: {stats.get('avg_relations_per_text', 0)}")
    
    # Local storage
    local_storage = results.get('local_storage', {})
    print(f"\n💾 Local Storage:")
    print(f"   • Raw Texts File: {local_storage.get('raw_texts_file', 'N/A')}")
    print(f"   • Extracted Knowledge: {local_storage.get('extracted_knowledge_file', 'N/A')}")
    print(f"   • Comparison File: {local_storage.get('comparison_file', 'N/A')}")
    
    # Azure storage
    azure_storage = results.get('azure_storage', {})
    print(f"\n☁️  Azure Storage:")
    print(f"   • Blob Container: {azure_storage.get('blob_container', 'N/A')}")
    print(f"   • Search Index: {azure_storage.get('search_index', 'N/A')}")
    print(f"   • Cosmos Container: {azure_storage.get('cosmos_container', 'N/A')}")
    print(f"   • Upload Status: {azure_storage.get('upload_status', 'N/A')}")
    
    # File locations
    file_paths = summary.get('file_paths', {})
    print(f"\n📂 File Locations:")
    print(f"   • Output Directory: {file_paths.get('local_output_directory', 'N/A')}")
    print(f"   • Batch Directory: {file_paths.get('batch_directory', 'N/A')}")
    print(f"   • Master Results: {file_paths.get('master_results', 'N/A')}")


def show_sample_extractions(results: dict, num_samples: int = 3):
    """Show sample extraction results"""
    
    print(f"\n{'='*80}")
    print(f"🔍 SAMPLE EXTRACTION RESULTS (First {num_samples})")
    print(f"{'='*80}")
    
    extractions = results.get('extraction_results', [])
    
    for i, extraction in enumerate(extractions[:num_samples]):
        print(f"\n📝 TEXT {i+1} (ID: {extraction.get('text_id', 'N/A')}):")
        print(f"   Raw: \"{extraction.get('raw_text', 'N/A')}\"")
        
        entities = extraction.get('entities', [])
        print(f"   🏷️  Entities ({len(entities)}):")
        for entity in entities:
            print(f"      • {entity.get('text', 'N/A')} [{entity.get('entity_type', 'unknown')}]")
        
        relations = extraction.get('relations', [])
        print(f"   🔗 Relations ({len(relations)}):")
        for relation in relations:
            source_id = relation.get('source_entity_id', '')
            target_id = relation.get('target_entity_id', '')
            
            # Find entity texts
            source_text = next((e['text'] for e in entities if e['entity_id'] == source_id), source_id)
            target_text = next((e['text'] for e in entities if e['entity_id'] == target_id), target_id)
            
            print(f"      • {source_text} --[{relation.get('relation_type', 'unknown')}]--> {target_text}")


def verify_local_files(results: dict):
    """Verify local files were created"""
    
    print(f"\n{'='*80}")
    print("🔍 LOCAL FILE VERIFICATION")
    print(f"{'='*80}")
    
    file_paths = results.get('storage_summary', {}).get('file_paths', {})
    
    # Check master results file
    master_file = Path(file_paths.get('master_results', ''))
    if master_file.exists():
        print(f"✅ Master Results: {master_file} ({master_file.stat().st_size} bytes)")
        
        # Show sample content
        with open(master_file, 'r') as f:
            content = f.read()
            print(f"   Sample content (first 200 chars): {content[:200]}...")
    else:
        print(f"❌ Master Results: File not found")
    
    # Check batch directory
    batch_dir = Path(file_paths.get('batch_directory', ''))
    if batch_dir.exists():
        files = list(batch_dir.glob('*'))
        print(f"✅ Batch Directory: {batch_dir} ({len(files)} files)")
        
        # List some files
        for file_path in files[:5]:  # Show first 5 files
            print(f"   • {file_path.name} ({file_path.stat().st_size} bytes)")
        
        if len(files) > 5:
            print(f"   ... and {len(files) - 5} more files")
    else:
        print(f"❌ Batch Directory: Not found")
    
    # Check individual data files
    local_storage = results.get('local_storage', {})
    for file_type, file_path in local_storage.items():
        if file_path and file_path != 'N/A':
            path_obj = Path(file_path)
            if path_obj.exists():
                print(f"✅ {file_type}: {path_obj.name} ({path_obj.stat().st_size} bytes)")
            else:
                print(f"❌ {file_type}: File not found")


def main():
    """Run dual storage extraction test"""
    
    print("🔄 DUAL STORAGE KNOWLEDGE EXTRACTION TEST")
    print("=" * 60)
    print("This test extracts knowledge and stores results BOTH locally and in Azure services")
    print("(Azure services are mocked for this demonstration)")
    
    # Load sample texts
    sample_texts = load_sample_texts(limit=10)
    print(f"\n✅ Loaded {len(sample_texts)} sample maintenance texts")
    
    # Initialize dual storage extractor
    print(f"\n🚀 Initializing DualStorageExtractor...")
    extractor = DualStorageExtractor("maintenance")
    
    # Run extraction with dual storage
    batch_name = f"demo_batch_{len(sample_texts)}_texts"
    print(f"\n🔍 Running dual storage extraction...")
    print(f"Batch name: {batch_name}")
    print(f"This will show real-time extraction and store results in multiple locations.\n")
    
    try:
        results = extractor.extract_and_store_all(sample_texts, batch_name)
        
        # Display comprehensive results
        display_storage_results(results)
        
        # Show sample extractions
        show_sample_extractions(results, num_samples=3)
        
        # Verify local files were created
        verify_local_files(results)
        
        print(f"\n{'='*80}")
        print("🎯 DUAL STORAGE TEST COMPLETE")
        print(f"{'='*80}")
        
        print(f"\n📁 Results Location:")
        output_dir = results.get('storage_summary', {}).get('file_paths', {}).get('local_output_directory', 'N/A')
        print(f"   Local Output Directory: {output_dir}")
        print(f"   Master Results File: {results.get('storage_summary', {}).get('file_paths', {}).get('master_results', 'N/A')}")
        
        print(f"\n💡 What was created:")
        print(f"   ✅ Individual text files (raw + extracted + comparison)")
        print(f"   ✅ Batch summary files (JSONL format)")
        print(f"   ✅ Master results file (complete analysis)")
        print(f"   ✅ Mock Azure storage (blob, search, cosmos)")
        
        print(f"\n🔍 To explore results:")
        print(f"   • Check the output directory: {output_dir}")
        print(f"   • View individual comparisons in markdown format")
        print(f"   • Examine JSONL files for programmatic access")
        print(f"   • Review master results for complete overview")
        
    except Exception as e:
        print(f"❌ Dual storage test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
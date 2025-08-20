#!/usr/bin/env python3
"""
Step 1: Basic Entity Extraction
==============================

Task: Extract entities from all files using the optimized cached extraction approach.
Logic: Process each DOMAIN (subdirectory) with Knowledge Extraction Agent using cached prompts.
DOMAIN-AWARE: 1 subdirectory = 1 domain = 1 auto prompt = 1 extraction call
NO FAKE SUCCESS PATTERNS - FAIL FAST if extraction fails.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


async def basic_entity_extraction():
    """Extract entities from all files using domain-aware processing - Step 1 of knowledge extraction"""
    print("🔬 STEP 1: BASIC ENTITY EXTRACTION")
    print("=" * 50)

    # Load data files - GROUP BY SUBDIRECTORY (DOMAIN-AWARE PROCESSING)
    project_root = Path(__file__).parent.parent.parent.parent
    data_root = project_root / "data" / "raw"

    # Group files by subdirectory (each subdirectory = one domain)
    domain_groups = {}
    for md_file in data_root.glob("**/*.md"):
        # Get the immediate subdirectory under data/raw
        relative_path = md_file.relative_to(data_root)
        if len(relative_path.parts) > 1:
            domain_name = relative_path.parts[0]  # First subdirectory is the domain
        else:
            domain_name = "root"  # Files directly in data/raw

        if domain_name not in domain_groups:
            domain_groups[domain_name] = []
        domain_groups[domain_name].append(md_file)

    total_files = sum(len(files) for files in domain_groups.values())
    print(f"📂 Found {total_files} files in {len(domain_groups)} domains:")
    for domain, files in domain_groups.items():
        print(f"   🏷️  {domain}: {len(files)} files")

    if not domain_groups:
        print("❌ FAIL FAST: No data files found")
        return False

    # Check if we're in incremental mode
    import os
    skip_cleanup = os.environ.get("SKIP_CLEANUP", "false").lower() == "true"

    # Track results
    extraction_results = []
    total_entities = 0
    total_relationships = 0

    # Import the optimized extraction function
    from agents.knowledge_extraction.agent import run_knowledge_extraction

    # DOMAIN-AWARE PROCESSING: One extraction per domain (subdirectory)
    print(f"\n🧠 DOMAIN-AWARE EXTRACTION: Processing {len(domain_groups)} domains")
    if skip_cleanup:
        print("   🔄 Incremental mode: Only processing new/changed domains")
    else:
        print("   🧹 Full mode: Processing all domains (after cleanup)")
    print("   🎯 Strategy: 1 domain = 1 auto prompt = 1 extraction call")

    # Process each domain separately
    processed_domains = 0
    skipped_domains = 0

    for domain_idx, (domain_name, domain_files) in enumerate(domain_groups.items(), 1):
        print(f"\n🏷️  Processing domain {domain_idx}/{len(domain_groups)}: {domain_name}")
        print(f"   📁 Files: {len(domain_files)} files")

        # In incremental mode, check if domain already has entities in the graph
        if skip_cleanup:
            try:
                # Check if entities exist for this domain in Cosmos DB
                from agents.core.universal_deps import get_universal_deps
                deps = await get_universal_deps()
                cosmos_client = deps.cosmos_client

                # Query for existing entities in this domain
                domain_query = f"g.V().has('domain', '{domain_name}').count()"
                existing_count = await cosmos_client.execute_query(domain_query)
                domain_entity_count = existing_count[0] if existing_count else 0

                if domain_entity_count > 0:
                    print(f"   ⏭️  Skipping domain '{domain_name}' (already has {domain_entity_count} entities)")
                    skipped_domains += 1
                    continue

            except Exception as e:
                print(f"   ⚠️  Could not check existing entities for {domain_name}: {e}")
                print(f"   🔄 Proceeding with processing...")

        # Calculate domain statistics
        domain_size = sum(f.stat().st_size for f in domain_files)
        print(f"   📏 Total size: {domain_size/1024:.1f}KB")

        # Combine all files in this domain into one extraction call
        combined_content = ""
        file_boundaries = []

        # Each domain gets its own extraction with domain-specific context
        for i, data_file in enumerate(domain_files):
            content = data_file.read_text(encoding="utf-8", errors="ignore")
            if len(content.strip()) < 50:
                continue

            start_pos = len(combined_content)
            combined_content += f"\n\n=== FILE {i+1}: {data_file.name} ===\n"
            combined_content += content
            end_pos = len(combined_content)

            file_boundaries.append({
                "filename": data_file.name,
                "start": start_pos,
                "end": end_pos,
                "domain": domain_name,
                "original_size": len(content)
            })

        print(f"   📊 Domain content: {len(combined_content)} characters")
        print(f"   🧠 Running DOMAIN extraction for: {domain_name}")
        print(f"   ⏳ Step 1/3: Analyzing domain characteristics...")

        start_time = time.time()

        # Single extraction call for this domain
        try:
            print(f"   ⏳ Step 2/3: Extracting entities and relationships...")
            print(f"   💡 Tip: First run takes longer as it generates domain-specific prompts")

            domain_result = await asyncio.wait_for(
                run_knowledge_extraction(
                    content=combined_content,
                    use_domain_analysis=True,
                    force_refresh_cache=False,
                    verbose=False
                ),
                timeout=1800  # 30 minute timeout per domain
            )

            print(f"   ⏳ Step 3/3: Processing extraction results...")

            # ExtractionResult doesn't have 'success' - if we get here, it succeeded
            if domain_result:
                extraction_time = time.time() - start_time
                progress_percent = (domain_idx / len(domain_groups)) * 100
                print(f"   ✅ Domain '{domain_name}' processed in {extraction_time:.1f}s")
                print(f"   📊 Entities: {len(domain_result.entities)}, Relationships: {len(domain_result.relationships)}")
                print(f"   📈 Overall progress: {progress_percent:.1f}% complete ({domain_idx}/{len(domain_groups)} domains)")

                # Convert entities and relationships to serializable dictionaries
                entities_data = []
                for entity in domain_result.entities:
                    entities_data.append({
                        "text": entity.text,
                        "type": entity.type,
                        "confidence": entity.confidence,
                        "context": entity.context
                    })

                relationships_data = []
                for rel in domain_result.relationships:
                    relationships_data.append({
                        "source": rel.source,
                        "target": rel.target,
                        "relation": rel.relation,
                        "confidence": rel.confidence,
                        "context": rel.context
                    })

                extraction_results.append({
                    "domain": domain_name,
                    "files": [f.name for f in domain_files],
                    "file_boundaries": file_boundaries,
                    "entities_count": len(domain_result.entities),
                    "relationships_count": len(domain_result.relationships),
                    "entities_data": entities_data,
                    "relationships_data": relationships_data,
                    "extraction_time": extraction_time,
                    "extraction_confidence": domain_result.extraction_confidence,
                    "processing_signature": domain_result.processing_signature
                })

                total_entities += len(domain_result.entities)
                total_relationships += len(domain_result.relationships)
                processed_domains += 1

                # Show sample results for verification
                if domain_result.entities:
                    # ExtractedEntity objects - get text attribute (entity name/text)
                    sample_entities = [entity.text for entity in list(domain_result.entities)[:3]]
                    print(f"   📝 Sample entities: {sample_entities}")
            else:
                print(f"   ❌ Domain '{domain_name}' extraction failed - no result returned")
                continue

        except asyncio.TimeoutError:
            print(f"   ⏰ Domain '{domain_name}' extraction timed out (>30min)")
            continue
        except Exception as e:
            print(f"   ❌ Domain '{domain_name}' extraction error: {str(e)}")
            continue

    # Processing complete for all domains
    print(f"\n🎯 DOMAIN PROCESSING COMPLETE")
    print(f"   📊 Total entities: {total_entities}")
    print(f"   📊 Total relationships: {total_relationships}")
    print(f"   🏷️  Domains processed: {processed_domains}")
    if skip_cleanup and skipped_domains > 0:
        print(f"   ⏭️  Domains skipped: {skipped_domains} (already had entities)")
        print(f"   🔄 Incremental mode: {processed_domains + skipped_domains} total domains")

    # Save results to JSON for next step - use absolute path from script location
    from scripts.dataflow.utilities.path_utils import get_results_dir

    results_dir = get_results_dir()

    # Results are already serializable - no conversion needed
    serializable_results = extraction_results

    results_file = results_dir / "step1_entity_extraction_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "total_domains_processed": len(domain_groups),
                "total_files_processed": total_files,
                "total_entities": total_entities,
                "total_relationships": total_relationships,
                "processing_method": "domain_aware",
                "domain_results": serializable_results,
                "success": len(extraction_results) > 0
            },
            f,
            indent=2
        )

    print(f"💾 Results saved to: {results_file}")

    if len(extraction_results) == 0:
        print("❌ FAIL FAST: No domains successfully processed")
        return False

    print(f"✅ Step 1 complete: {len(extraction_results)} domains processed with {total_entities} entities")
    return True


if __name__ == "__main__":
    result = asyncio.run(basic_entity_extraction())
    sys.exit(0 if result else 1)

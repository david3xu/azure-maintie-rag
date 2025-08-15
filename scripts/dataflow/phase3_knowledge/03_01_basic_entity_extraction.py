#!/usr/bin/env python3
"""
Step 1: Basic Entity Extraction
==============================

Task: Extract entities from all files using the optimized cached extraction approach.
Logic: Process each file with Knowledge Extraction Agent using cached prompts.
NO FAKE SUCCESS PATTERNS - FAIL FAST if extraction fails.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


async def basic_entity_extraction():
    """Extract entities from all files using cached prompts - Step 1 of knowledge extraction"""
    print("🔬 STEP 1: BASIC ENTITY EXTRACTION")
    print("=" * 50)

    # Load data files - DISCOVER files dynamically (no hardcoded paths)
    project_root = Path(__file__).parent.parent.parent.parent
    data_root = project_root / "data" / "raw"
    data_files = list(data_root.glob("**/*.md"))  # Find all .md files recursively

    print(f"📂 Found {len(data_files)} files to process")

    if not data_files:
        print("❌ FAIL FAST: No data files found")
        return False

    # Track results
    extraction_results = []
    total_entities = 0
    total_relationships = 0

    # Import the optimized extraction function
    from agents.knowledge_extraction.agent import run_knowledge_extraction

    # Process each file with timeout and error handling
    for i, data_file in enumerate(data_files, 1):
        try:
            content = data_file.read_text(encoding="utf-8", errors="ignore")
            print(f"\n📄 Processing {i}/{len(data_files)}: {data_file.name}")
            print(f"   📊 Content: {len(content)} characters")

            if len(content.strip()) < 100:  # Skip very small files
                print(f"   ⚠️  Skipping: File too small (< 100 chars)")
                continue

            # Limit content size to avoid timeouts (use first portion)
            content_chunk = content[:1500] if len(content) > 1500 else content
            print(f"   🔄 Processing chunk: {len(content_chunk)} characters")

            # Use the optimized cached extraction with timeout
            print(f"   🧠 Running knowledge extraction...")
            start_time = time.time()

            result = await asyncio.wait_for(
                run_knowledge_extraction(
                    content=content_chunk,
                    use_domain_analysis=True,  # Use cached auto prompts
                    force_refresh_cache=False,  # Reuse cache when possible
                    verbose=False,  # Reduce output for pipeline
                ),
                timeout=60,  # 60 second timeout per file
            )

            extraction_time = time.time() - start_time

            # Validate results
            entities_count = len(result.entities)
            relationships_count = len(result.relationships)

            print(
                f"   ✅ Extracted: {entities_count} entities, {relationships_count} relationships"
            )
            print(f"   ⏱️  Time: {extraction_time:.2f}s")
            print(f"   🎯 Confidence: {result.extraction_confidence:.3f}")

            # Store results for next step
            file_result = {
                "filename": data_file.name,
                "filepath": str(data_file),
                "entities_count": entities_count,
                "relationships_count": relationships_count,
                "extraction_time": extraction_time,
                "confidence": result.extraction_confidence,
                "processing_signature": result.processing_signature,
                "extraction_result": result,  # Keep full result for storage step
            }

            extraction_results.append(file_result)
            total_entities += entities_count
            total_relationships += relationships_count

            # Show sample entities if any
            if result.entities:
                sample_entities = [e.text for e in result.entities[:3]]
                print(f"   📝 Sample entities: {sample_entities}")

        except asyncio.TimeoutError:
            print(f"   ❌ TIMEOUT: File {data_file.name} exceeded 60s limit")
            continue
        except Exception as e:
            print(f"   ❌ ERROR: {data_file.name}: {e}")
            continue

    # Save results to JSON for next step - use absolute path from script location
    script_dir = Path(__file__).parent.parent.parent  # Go up to project root
    results_dir = script_dir / "scripts" / "dataflow" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)  # Create parent directories if needed

    # Prepare serializable results (without the full extraction_result object)
    serializable_results = []
    for result in extraction_results:
        serializable_result = result.copy()
        serializable_result.pop("extraction_result")  # Remove non-serializable object
        serializable_results.append(serializable_result)

    results_file = results_dir / "step1_entity_extraction_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "total_files_processed": len(data_files),
                "successful_extractions": len(extraction_results),
                "total_entities": total_entities,
                "total_relationships": total_relationships,
                "results": serializable_results,
            },
            f,
            indent=2,
        )

    # Final results
    print(f"\n📊 STEP 1 EXTRACTION RESULTS:")
    print(f"=" * 40)
    print(
        f"✅ Successfully processed: {len(extraction_results)}/{len(data_files)} files"
    )
    print(f"📊 Total entities extracted: {total_entities}")
    print(f"📊 Total relationships extracted: {total_relationships}")
    print(f"💾 Results saved to: {results_file}")

    if len(extraction_results) > 0 and total_entities > 0:
        avg_entities = total_entities / len(extraction_results)
        avg_relationships = total_relationships / len(extraction_results)
        print(f"📈 Average entities per file: {avg_entities:.1f}")
        print(f"📈 Average relationships per file: {avg_relationships:.1f}")
        print(f"\n🎉 STEP 1 COMPLETED SUCCESSFULLY")
        print(f"🔄 Ready for Step 2: Storage in graph database")
        return True
    elif len(extraction_results) > 0 and total_entities == 0:
        print(f"\n❌ FAIL FAST: Files processed but NO ENTITIES EXTRACTED")
        print(f"💡 Check extraction prompts and Azure OpenAI responses")
        return False
    else:
        print(f"\n❌ FAIL FAST: No files were processed successfully")
        return False


async def main():
    """Main execution function"""
    success = await basic_entity_extraction()
    return success


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        if result:
            print("\n✅ Step 1: Basic entity extraction completed")
            sys.exit(0)
        else:
            print("\n❌ Step 1: Basic entity extraction failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Step 1 error: {e}")
        sys.exit(1)

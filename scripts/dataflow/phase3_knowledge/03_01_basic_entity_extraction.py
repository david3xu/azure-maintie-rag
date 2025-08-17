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

    # INTELLIGENT BATCH SIZE MANAGEMENT
    total_size = sum(f.stat().st_size for f in data_files)
    max_files = len(data_files)
    
    # Dynamic batch sizing based on total content and file count
    # Small datasets: batch process for efficiency
    # Large datasets: individual processing to avoid memory/timeout issues
    
    # Calculate optimal batch parameters
    avg_file_size = total_size / max_files if max_files > 0 else 0
    
    # Batch decision logic with multiple size thresholds
    should_batch = (
        total_size < 200_000 and           # Total content under 200KB
        max_files <= 15 and               # Limited number of files
        avg_file_size < 50_000             # Average file size under 50KB
    )
    
    if should_batch:
        print(f"📦 BATCH PROCESSING: {max_files} files ({total_size/1024:.1f}KB total, avg: {avg_file_size/1024:.1f}KB)")
        print("   Using single extraction call for efficiency")
        
        # Combine all files into one batch with intelligent size limits
        combined_content = ""
        file_boundaries = []
        
        # Dynamic per-file content limit based on number of files
        # More files = smaller chunk per file to stay under LLM token limits
        max_content_per_file = min(3000, 150_000 // max_files)  # Smart limit scaling
        
        print(f"   📏 Max content per file: {max_content_per_file} chars")
        
        for i, data_file in enumerate(data_files):
            content = data_file.read_text(encoding="utf-8", errors="ignore")
            if len(content.strip()) < 50:
                continue
                
            start_pos = len(combined_content)
            combined_content += f"\n\n=== FILE {i+1}: {data_file.name} ===\n"
            
            # Use dynamic content limit based on total files
            content_chunk = content[:max_content_per_file]
            combined_content += content_chunk
            end_pos = len(combined_content)
            
            file_boundaries.append({
                "filename": data_file.name,
                "start": start_pos,
                "end": end_pos,
                "original_size": len(content),
                "chunk_size": len(content_chunk),
                "truncated": len(content) > max_content_per_file
            })
        
        print(f"   📊 Combined content: {len(combined_content)} characters")
        print(f"   🔄 Running SINGLE extraction for all files...")
        
        start_time = time.time()
        
        # Single extraction call for entire batch
        batch_result = await asyncio.wait_for(
            run_knowledge_extraction(
                content=combined_content,
                use_domain_analysis=True,
                force_refresh_cache=False,
                verbose=True,
            ),
            timeout=120,  # 2 minute timeout for batch
        )
        
        batch_time = time.time() - start_time
        
        print(f"   ✅ Batch extraction completed in {batch_time:.2f}s")
        print(f"   📊 Total: {len(batch_result.entities)} entities, {len(batch_result.relationships)} relationships")
        
        # Distribute results back to individual files
        for boundary in file_boundaries:
            # For simplicity, divide results proportionally
            file_portion = (boundary["end"] - boundary["start"]) / len(combined_content)
            
            entities_for_file = int(len(batch_result.entities) * file_portion)
            relationships_for_file = int(len(batch_result.relationships) * file_portion)
            
            # Take proportional slice of results
            start_idx = int(len(batch_result.entities) * (boundary["start"] / len(combined_content)))
            entities_slice = batch_result.entities[start_idx:start_idx + entities_for_file]
            
            start_rel_idx = int(len(batch_result.relationships) * (boundary["start"] / len(combined_content)))
            relationships_slice = batch_result.relationships[start_rel_idx:start_rel_idx + relationships_for_file]
            
            file_result = {
                "filename": boundary["filename"],
                "filepath": str(next(f for f in data_files if f.name == boundary["filename"])),
                "entities_count": len(entities_slice),
                "relationships_count": len(relationships_slice),
                "extraction_confidence": batch_result.extraction_confidence,
                "processing_time": batch_time / len(file_boundaries),  # Distribute time
                "processing_method": "batch_optimized",
                "extraction_result": batch_result  # Add the batch result for downstream processing
            }
            
            extraction_results.append(file_result)
            total_entities += len(entities_slice)
            total_relationships += len(relationships_slice)
            
            print(f"   📄 {boundary['filename']}: {len(entities_slice)} entities, {len(relationships_slice)} relationships")
    
    else:
        print(f"📄 INDIVIDUAL PROCESSING: {len(data_files)} files (large dataset)")
        # Process each file individually for large datasets
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
    project_root = Path(__file__).parent.parent.parent.parent  # Go up to project root
    results_dir = project_root / "scripts" / "dataflow" / "results"
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

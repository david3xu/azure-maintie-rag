#!/usr/bin/env python3
"""
Knowledge Extraction - Stage 2 of Data Flow
Raw Text Data → Knowledge Extraction (Universal Agent)

This script implements the second stage of the data processing pipeline:
- Data-driven knowledge extraction without predetermined domain knowledge
- Uses Universal Agent for intelligent entity and relationship extraction
- Processes ingested data from previous stage
- Prepares structured knowledge for graph construction
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.universal_agent import universal_agent
from services.infrastructure_service import AsyncInfrastructureService

logger = logging.getLogger(__name__)


class KnowledgeExtractionStage:
    """Stage 2: Data-driven Knowledge Extraction using Universal Agent"""

    def __init__(self):
        self.infrastructure = AsyncInfrastructureService()

    async def execute(self, source_path: str) -> Dict[str, Any]:
        """
        Execute knowledge extraction stage using Universal Agent
        
        Data-driven approach - extracts knowledge without predetermined schemas

        Args:
            source_path: Path to processed data from previous stage

        Returns:
            Dict with extraction results and metadata
        """
        print("🧠 Stage 2: Knowledge Extraction - Text → Structured Knowledge")
        print("=" * 65)

        start_time = asyncio.get_event_loop().time()

        results = {
            "stage": "02_knowledge_extraction",
            "source_path": str(source_path),
            "success": False,
            "data_driven": True,
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "knowledge_domains": []
        }

        try:
            # Find files to process (from data ingestion output or raw data)
            source_directory = Path(source_path)
            if not source_directory.exists():
                raise FileNotFoundError(f"Source path does not exist: {source_path}")

            # Find all .md files recursively (data-driven discovery)
            md_files = list(source_directory.glob("**/*.md"))
            
            if not md_files:
                raise ValueError(f"No .md files found in {source_path}")

            print(f"📁 Found {len(md_files)} files for knowledge extraction")

            # Process each file for knowledge extraction
            extracted_knowledge = []
            total_entities = 0
            total_relationships = 0
            discovered_domains = set()

            for file_path in md_files:
                try:
                    print(f"🧠 Extracting knowledge from: {file_path.name}")
                    
                    # Read file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Extract knowledge using Universal Agent
                    knowledge_result = await self._extract_knowledge_with_agent(
                        file_path.name, content
                    )

                    if knowledge_result.get("success", False):
                        extracted_knowledge.append(knowledge_result)
                        
                        # Aggregate metrics
                        entities = knowledge_result.get("entities", [])
                        relationships = knowledge_result.get("relationships", [])
                        domains = knowledge_result.get("domains", [])
                        
                        total_entities += len(entities)
                        total_relationships += len(relationships)
                        discovered_domains.update(domains)
                        
                        print(f"✅ Extracted: {len(entities)} entities, {len(relationships)} relationships")
                    else:
                        print(f"⚠️ Partial extraction from: {file_path.name}")

                except Exception as e:
                    logger.error(f"Failed to extract knowledge from {file_path}: {e}")
                    print(f"❌ Failed to process: {file_path.name}")

            # Save extracted knowledge
            output_file = Path("data/processed/knowledge_extraction_results.json")
            await self._save_extracted_knowledge(extracted_knowledge, output_file)

            # Calculate success metrics
            success_rate = len(extracted_knowledge) / len(md_files) if md_files else 0

            results.update({
                "files_processed": len(md_files),
                "successful_extractions": len(extracted_knowledge),
                "success_rate": round(success_rate, 2),
                "entities_extracted": total_entities,
                "relationships_extracted": total_relationships,
                "knowledge_domains": list(discovered_domains),
                "extracted_knowledge": extracted_knowledge,
                "output_file": str(output_file),
                "success": success_rate > 0.5
            })

            # Duration
            duration = asyncio.get_event_loop().time() - start_time
            results["duration_seconds"] = round(duration, 2)

            print(f"\n📊 Knowledge Extraction Results:")
            print(f"   📁 Files processed: {len(extracted_knowledge)}/{len(md_files)}")
            print(f"   📈 Success rate: {results['success_rate']*100:.1f}%")
            print(f"   🎯 Entities extracted: {total_entities}")
            print(f"   🔗 Relationships found: {total_relationships}")
            print(f"   🏷️ Knowledge domains: {len(discovered_domains)}")
            print(f"   ⏱️  Duration: {results['duration_seconds']}s")

            if results["success"]:
                print("✅ Stage 2 Complete - Knowledge successfully extracted")
            else:
                print("⚠️ Stage 2 Partial - Some files failed to process")

            return results

        except Exception as e:
            results["error"] = str(e)
            results["duration_seconds"] = round(
                asyncio.get_event_loop().time() - start_time, 2
            )
            print(f"❌ Stage 2 Failed: {e}")
            logger.error(f"Knowledge extraction failed: {e}", exc_info=True)
            return results

    async def _extract_knowledge_with_agent(self, filename: str, content: str) -> Dict[str, Any]:
        """
        Extract structured knowledge using Universal Agent
        Data-driven approach - no predetermined schemas
        """
        try:
            # Create a knowledge extraction prompt for the Universal Agent
            # This is data-driven - agent determines the knowledge structure
            query = f"""
            Extract structured knowledge from this document content:
            
            Filename: {filename}
            Content length: {len(content)} characters
            
            Please extract:
            1. Key entities (people, organizations, technologies, concepts, etc.)
            2. Relationships between entities
            3. Domain categories this content belongs to
            4. Important facts and definitions
            
            Provide the response in a structured format with:
            - entities: list of important entities found
            - relationships: list of relationships between entities
            - domains: list of knowledge domains/categories
            - key_facts: list of important facts or definitions
            
            Content to analyze:
            {content[:3000]}...
            """

            # Run agent analysis
            agent_response = await universal_agent.run(query)
            agent_output = str(agent_response.output) if hasattr(agent_response, 'output') else str(agent_response)
            
            # Parse agent response for structured knowledge
            knowledge = await self._parse_agent_knowledge_response(agent_output, filename, content)
            
            print(f"🧠 Knowledge extraction completed for {filename}")
            
            return {
                "success": True,
                "filename": filename,
                "agent_response": agent_output[:500] + "..." if len(agent_output) > 500 else agent_output,
                "entities": knowledge.get("entities", []),
                "relationships": knowledge.get("relationships", []),
                "domains": knowledge.get("domains", []),
                "key_facts": knowledge.get("key_facts", []),
                "content_length": len(content),
                "data_driven_extraction": True
            }

        except Exception as e:
            logger.error(f"Agent knowledge extraction failed for {filename}: {e}")
            return {
                "success": False,
                "filename": filename,
                "error": str(e),
                "entities": [],
                "relationships": [],
                "domains": [],
                "key_facts": []
            }

    async def _parse_agent_knowledge_response(self, agent_output: str, filename: str, content: str) -> Dict[str, Any]:
        """
        Parse the agent's response to extract structured knowledge
        Implements fallback extraction if agent response isn't perfectly structured
        """
        try:
            # Try to extract structured information from agent response
            entities = []
            relationships = []
            domains = []
            key_facts = []
            
            # Simple parsing - look for common patterns in agent responses
            lines = agent_output.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Detect sections
                if 'entities' in line.lower() or 'entity' in line.lower():
                    current_section = 'entities'
                elif 'relationships' in line.lower() or 'relationship' in line.lower():
                    current_section = 'relationships'
                elif 'domains' in line.lower() or 'domain' in line.lower() or 'categories' in line.lower():
                    current_section = 'domains'
                elif 'facts' in line.lower() or 'definitions' in line.lower():
                    current_section = 'key_facts'
                elif line.startswith('- ') or line.startswith('* ') or line.startswith('• '):
                    # Extract list items
                    item = line[2:].strip()
                    if current_section == 'entities' and item:
                        entities.append(item)
                    elif current_section == 'relationships' and item:
                        relationships.append(item)
                    elif current_section == 'domains' and item:
                        domains.append(item)
                    elif current_section == 'key_facts' and item:
                        key_facts.append(item)

            # Fallback: basic extraction if structured parsing didn't work well
            if not entities and not relationships:
                # Use basic heuristics to extract some knowledge
                entities = await self._extract_basic_entities(content, filename)
                domains = ["technology", "documentation"]  # Basic domains
                
            return {
                "entities": entities[:20],  # Limit to top 20 entities
                "relationships": relationships[:15],  # Limit to top 15 relationships
                "domains": domains[:5],  # Limit to top 5 domains
                "key_facts": key_facts[:10]  # Limit to top 10 facts
            }

        except Exception as e:
            logger.warning(f"Failed to parse agent response for {filename}: {e}")
            # Return basic fallback extraction
            return {
                "entities": await self._extract_basic_entities(content, filename),
                "relationships": [],
                "domains": ["general"],
                "key_facts": []
            }

    async def _extract_basic_entities(self, content: str, filename: str) -> List[str]:
        """
        Basic entity extraction as fallback
        """
        try:
            # Simple entity extraction - look for capitalized words/phrases
            import re
            
            # Find capitalized words (potential entities)
            capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
            
            # Find technical terms (words with specific patterns)
            technical_terms = re.findall(r'\b[A-Z]{2,}\b', content)  # Acronyms
            
            # Combine and deduplicate
            entities = list(set(capitalized_words + technical_terms))
            
            # Filter out common words
            common_words = {'The', 'This', 'That', 'Azure', 'Microsoft', 'API', 'You', 'We', 'They'}
            entities = [e for e in entities if e not in common_words and len(e) > 2]
            
            return entities[:15]  # Return top 15 entities
            
        except Exception as e:
            logger.warning(f"Basic entity extraction failed for {filename}: {e}")
            return ["Document", filename.replace('.md', '')]

    async def _save_extracted_knowledge(self, extracted_knowledge: List[Dict], output_file: Path):
        """Save extracted knowledge to JSON file"""
        try:
            # Ensure directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare summary data
            summary_data = {
                "timestamp": asyncio.get_event_loop().time(),
                "total_files": len(extracted_knowledge),
                "extraction_method": "universal_agent_data_driven",
                "extracted_knowledge": extracted_knowledge
            }
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
                
            print(f"💾 Knowledge saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save extracted knowledge: {e}")
            print(f"⚠️ Warning: Could not save knowledge to {output_file}")


async def main():
    """Main entry point for knowledge extraction stage"""
    parser = argparse.ArgumentParser(
        description="Stage 2: Data-Driven Knowledge Extraction - Text → Structured Knowledge"
    )
    parser.add_argument("--source", required=True, help="Path to source data files")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    # Execute stage with data-driven approach
    stage = KnowledgeExtractionStage()
    results = await stage.execute(source_path=args.source)

    # Save results if requested
    if args.output and results.get("success"):
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"📄 Results saved to: {args.output}")

    # Return appropriate exit code
    return 0 if results.get("success") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
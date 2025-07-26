"""
Azure Prompt Flow Integration Service
Bridges existing universal extraction system with centralized prompt flows
Maintains backward compatibility while enabling prompt flow benefits
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import tempfile
import subprocess
import os

from config.settings import settings
from .prompt_flow_monitoring import prompt_flow_monitor

logger = logging.getLogger(__name__)


class AzurePromptFlowIntegrator:
    """
    Integration service for Azure Prompt Flow with universal knowledge extraction
    Provides centralized prompt management while preserving universal principles
    """
    
    def __init__(self, domain_name: str = "general"):
        self.domain_name = domain_name
        self.flow_path = Path(__file__).parent.parent.parent / "prompt_flows" / "universal_knowledge_extraction"
        self.enable_prompt_flow = getattr(settings, 'enable_prompt_flow', False)
        self.fallback_to_legacy = getattr(settings, 'prompt_flow_fallback_enabled', True)
        
        # Prompt Flow configuration
        self.flow_config = {
            "flow_path": str(self.flow_path),
            "connection_name": "azure_openai_connection",
            "runtime": "automatic",
            "environment_variables": {
                "AZURE_OPENAI_DEPLOYMENT_NAME": settings.openai_deployment_name,
                "AZURE_OPENAI_API_KEY": settings.openai_api_key,
                "AZURE_OPENAI_ENDPOINT": settings.openai_api_base,
                "AZURE_OPENAI_API_VERSION": settings.openai_api_version
            }
        }
        
        logger.info(f"AzurePromptFlowIntegrator initialized - Flow enabled: {self.enable_prompt_flow}")
    
    async def extract_knowledge_with_prompt_flow(
        self,
        texts: List[str],
        max_entities: int = 50,
        confidence_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Extract knowledge using Azure Prompt Flow with centralized prompts
        
        Args:
            texts: List of text documents to process
            max_entities: Maximum entities to extract
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Extraction results in universal format
        """
        if not self.enable_prompt_flow:
            logger.info("Prompt Flow disabled - using legacy extraction")
            return await self._fallback_to_legacy_extraction(texts, max_entities, confidence_threshold)
        
        try:
            # Start monitoring
            execution_id = f"pf_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            prompt_flow_monitor.start_execution_tracking(execution_id, "universal_knowledge_extraction", self.domain_name)
            
            logger.info(f"Starting Prompt Flow extraction for {len(texts)} texts (ID: {execution_id})")
            
            # Prepare Prompt Flow inputs
            flow_inputs = {
                "texts": texts[:10],  # Limit for performance
                "domain_name": self.domain_name,
                "max_entities": max_entities,
                "confidence_threshold": confidence_threshold
            }
            
            # Execute Prompt Flow
            flow_results = await self._execute_prompt_flow(flow_inputs)
            
            if not flow_results.get("success", False):
                logger.warning("Prompt Flow execution failed - falling back to legacy")
                prompt_flow_monitor.end_execution_tracking(execution_id, "failed", error_message="Prompt Flow execution failed")
                return await self._fallback_to_legacy_extraction(texts, max_entities, confidence_threshold)
            
            # Transform results to universal format
            universal_results = self._transform_to_universal_format(flow_results)
            
            # End monitoring with success
            prompt_flow_monitor.end_execution_tracking(execution_id, "completed", universal_results)
            
            logger.info(f"Prompt Flow extraction completed: {len(universal_results.get('entities', []))} entities")
            
            return universal_results
            
        except Exception as e:
            logger.error(f"Prompt Flow extraction failed: {e}", exc_info=True)
            
            if self.fallback_to_legacy:
                logger.info("Falling back to legacy extraction system")
                return await self._fallback_to_legacy_extraction(texts, max_entities, confidence_threshold)
            else:
                return {
                    "success": False,
                    "error": f"Prompt Flow extraction failed: {e}",
                    "entities": [],
                    "relations": [],
                    "timestamp": datetime.now().isoformat()
                }
    
    async def _execute_prompt_flow(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Azure Prompt Flow with given inputs"""
        try:
            # Create temporary input file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(inputs, temp_file, indent=2)
                temp_input_path = temp_file.name
            
            # Create temporary output file path
            temp_output_path = temp_input_path.replace('.json', '_output.json')
            
            # Prepare Prompt Flow command
            pf_command = [
                "pf", "flow", "run",
                "--flow", str(self.flow_path),
                "--data", temp_input_path,
                "--output", temp_output_path
            ]
            
            # Set environment variables
            env = os.environ.copy()
            env.update(self.flow_config["environment_variables"])
            
            # Execute Prompt Flow
            logger.info(f"Executing Prompt Flow: {' '.join(pf_command)}")
            
            process = await asyncio.create_subprocess_exec(
                *pf_command,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Prompt Flow execution failed: {stderr.decode()}")
            
            # Read results
            if os.path.exists(temp_output_path):
                with open(temp_output_path, 'r') as f:
                    results = json.load(f)
            else:
                raise Exception("Prompt Flow output file not found")
            
            # Cleanup temporary files
            try:
                os.unlink(temp_input_path)
                os.unlink(temp_output_path)
            except OSError:
                pass
            
            return {
                "success": True,
                "results": results,
                "stdout": stdout.decode(),
                "execution_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prompt Flow execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": {}
            }
    
    async def _fallback_to_legacy_extraction(
        self,
        texts: List[str],
        max_entities: int,
        confidence_threshold: float
    ) -> Dict[str, Any]:
        """Fallback to legacy extraction system"""
        try:
            # Import legacy extractor
            from ..azure_openai.knowledge_extractor import AzureOpenAIKnowledgeExtractor
            
            extractor = AzureOpenAIKnowledgeExtractor(self.domain_name)
            results = await extractor.extract_knowledge_from_texts(texts)
            
            logger.info("Legacy extraction completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Legacy extraction fallback failed: {e}")
            return {
                "success": False,
                "error": f"Both Prompt Flow and legacy extraction failed: {e}",
                "entities": [],
                "relations": []
            }
    
    def _transform_to_universal_format(self, flow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Prompt Flow results to universal extraction format"""
        try:
            results = flow_results.get("results", {})
            
            # Extract components from Prompt Flow results
            entities = results.get("entities", [])
            relations = results.get("relations", [])
            summary = results.get("extraction_summary", {})
            
            # Build universal format
            universal_results = {
                "success": True,
                "domain": self.domain_name,
                "entities": entities,
                "relations": relations,
                "extraction_stats": {
                    "total_entities_extracted": len(entities),
                    "total_relations_extracted": len(relations),
                    "unique_entity_types": len(set(e.get("entity_type", "") for e in entities)),
                    "unique_relation_types": len(set(r.get("relation_type", "") for r in relations)),
                    "extraction_method": "azure_prompt_flow"
                },
                "knowledge_summary": summary.get("extraction_metrics", {}),
                "quality_assessment": summary,
                "prompt_flow_metadata": {
                    "flow_path": str(self.flow_path),
                    "execution_method": "centralized_prompts",
                    "universal_principles": True
                },
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("Results transformed to universal format")
            return universal_results
            
        except Exception as e:
            logger.error(f"Result transformation failed: {e}")
            return {
                "success": False,
                "error": f"Result transformation failed: {e}",
                "entities": [],
                "relations": []
            }
    
    def get_prompt_templates(self) -> Dict[str, str]:
        """Get centralized prompt templates for inspection/modification"""
        templates = {}
        
        try:
            # Read entity extraction template
            entity_template_path = self.flow_path / "entity_extraction.jinja2"
            if entity_template_path.exists():
                with open(entity_template_path, 'r') as f:
                    templates["entity_extraction"] = f.read()
            
            # Read relation extraction template
            relation_template_path = self.flow_path / "relation_extraction.jinja2"
            if relation_template_path.exists():
                with open(relation_template_path, 'r') as f:
                    templates["relation_extraction"] = f.read()
            
            logger.info(f"Retrieved {len(templates)} prompt templates")
            
        except Exception as e:
            logger.error(f"Failed to read prompt templates: {e}")
        
        return templates
    
    def update_prompt_template(self, template_name: str, template_content: str) -> bool:
        """Update centralized prompt template"""
        try:
            template_path = self.flow_path / f"{template_name}.jinja2"
            
            # Backup existing template
            if template_path.exists():
                backup_path = template_path.with_suffix(f".jinja2.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                template_path.rename(backup_path)
                logger.info(f"Backed up existing template to {backup_path}")
            
            # Write new template
            with open(template_path, 'w') as f:
                f.write(template_content)
            
            logger.info(f"Updated prompt template: {template_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update prompt template {template_name}: {e}")
            return False


# Convenience function for easy integration
async def extract_knowledge_with_centralized_prompts(
    texts: List[str],
    domain_name: str = "general",
    max_entities: int = 50,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Convenience function for knowledge extraction with centralized prompts
    
    Args:
        texts: List of text documents
        domain_name: Domain context (universal)
        max_entities: Maximum entities to extract
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        Universal extraction results
    """
    integrator = AzurePromptFlowIntegrator(domain_name)
    return await integrator.extract_knowledge_with_prompt_flow(
        texts, max_entities, confidence_threshold
    )


if __name__ == "__main__":
    # Test the integration
    async def test_integration():
        sample_texts = [
            "The hydraulic system contains multiple valves that control fluid flow.",
            "Regular bearing maintenance includes lubrication and inspection procedures.",
            "Engine sensors monitor temperature, pressure, and operational status."
        ]
        
        results = await extract_knowledge_with_centralized_prompts(sample_texts)
        print(json.dumps(results, indent=2))
    
    asyncio.run(test_integration())
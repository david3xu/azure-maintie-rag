"""
Simple MaintIE knowledge extraction - realistic approach
Extracts basic lists and patterns from MaintIE data without over-engineering
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

from src.models.maintenance_models import MaintenanceEntity, MaintenanceDocument, EntityType
from config.settings import settings


logger = logging.getLogger(__name__)


class SimpleMaintIEExtractor:
    """Extract basic lists from MaintIE data - realistic approach"""

    def __init__(self):
        self.config_path = Path("config/domain_knowledge.json")

    def extract_equipment_terms(self, entities: List[MaintenanceEntity]) -> List[str]:
        """Extract equipment terms from MaintIE entities"""
        equipment_terms = []

        for entity in entities:
            if entity.entity_type == EntityType.PHYSICAL_OBJECT:
                # Simple extraction - just get the text
                equipment_terms.append(entity.text.lower())

        # Remove duplicates and return most common
        return list(set(equipment_terms))

    def extract_common_abbreviations(self, documents: List[MaintenanceDocument]) -> Dict[str, str]:
        """Find abbreviations in MaintIE texts using simple pattern matching"""
        abbreviations = {}

        for doc in documents:
            # Simple regex to find patterns like "PM (preventive maintenance)"
            abbrev_pattern = r'(\b[A-Z]{2,5}\b)\s*\([^)]*([^)]+)\)'
            matches = re.findall(abbrev_pattern, doc.text)

            for abbrev, expansion in matches:
                abbreviations[abbrev.lower()] = expansion.lower().strip()

        return abbreviations

    def extract_failure_terms(self, documents: List[MaintenanceDocument]) -> List[str]:
        """Extract failure-related terms from document text"""
        failure_terms = []
        failure_keywords = ['failure', 'leak', 'vibration', 'noise', 'overheating', 'wear', 'corrosion', 'crack']

        for doc in documents:
            text_lower = doc.text.lower()
            for keyword in failure_keywords:
                if keyword in text_lower:
                    failure_terms.append(keyword)

        return list(set(failure_terms))

    def extract_procedure_terms(self, documents: List[MaintenanceDocument]) -> List[str]:
        """Extract procedure-related terms from document text"""
        procedure_terms = []
        procedure_keywords = ['maintenance', 'inspection', 'repair', 'replacement', 'installation', 'calibration', 'testing']

        for doc in documents:
            text_lower = doc.text.lower()
            for keyword in procedure_keywords:
                if keyword in text_lower:
                    procedure_terms.append(keyword)

        return list(set(procedure_terms))

    def update_domain_config(self) -> None:
        """Update domain config with extracted knowledge"""
        try:
            # Load MaintIE data
            entities = self._load_entities()
            documents = self._load_documents()

            if not entities and not documents:
                logger.warning("No MaintIE data found for extraction")
                return

            # Extract simple patterns
            equipment_terms = self.extract_equipment_terms(entities)
            abbreviations = self.extract_common_abbreviations(documents)
            failure_terms = self.extract_failure_terms(documents)
            procedure_terms = self.extract_procedure_terms(documents)

            # Update config file
            config = self._load_existing_config()

            # Update with extracted knowledge
            config["maintie_equipment"] = equipment_terms[:50]  # Top 50
            config["extracted_abbreviations"] = abbreviations
            config["extracted_failure_terms"] = failure_terms
            config["extracted_procedure_terms"] = procedure_terms

            # Save updated config
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"Updated domain config with {len(equipment_terms)} equipment terms, "
                       f"{len(abbreviations)} abbreviations, {len(failure_terms)} failure terms, "
                       f"{len(procedure_terms)} procedure terms")

        except Exception as e:
            logger.error(f"Error updating domain config: {e}")

    def _load_entities(self) -> List[MaintenanceEntity]:
        """Load entities from processed data"""
        try:
            entities_path = settings.processed_data_dir / "maintenance_entities.json"
            if not entities_path.exists():
                logger.warning(f"Entities file not found: {entities_path}")
                return []

            with open(entities_path, 'r', encoding='utf-8') as f:
                entities_data = json.load(f)

            entities = []
            for entity_data in entities_data:
                try:
                    entity = MaintenanceEntity.from_dict(entity_data)
                    entities.append(entity)
                except Exception as e:
                    logger.warning(f"Error loading entity: {e}")
                    continue

            logger.info(f"Loaded {len(entities)} entities from MaintIE data")
            return entities

        except Exception as e:
            logger.error(f"Error loading entities: {e}")
            return []

    def _load_documents(self) -> List[MaintenanceDocument]:
        """Load documents from processed data"""
        try:
            documents_path = settings.processed_data_dir / "maintenance_documents.json"
            if not documents_path.exists():
                logger.warning(f"Documents file not found: {documents_path}")
                return []

            with open(documents_path, 'r', encoding='utf-8') as f:
                docs_data = json.load(f)

            documents = []
            for doc_data in docs_data:
                try:
                    doc = MaintenanceDocument(
                        doc_id=doc_data["doc_id"],
                        text=doc_data["text"],
                        title=doc_data.get("title"),
                        metadata=doc_data.get("metadata", {})
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Error loading document: {e}")
                    continue

            logger.info(f"Loaded {len(documents)} documents from MaintIE data")
            return documents

        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []

    def _load_existing_config(self) -> Dict[str, Any]:
        """Load existing domain knowledge config"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Domain knowledge config not found: {self.config_path}")
                return self._get_minimal_defaults()
        except Exception as e:
            logger.error(f"Error loading domain config: {e}")
            return self._get_minimal_defaults()

    def _get_minimal_defaults(self) -> Dict[str, Any]:
        """Minimal fallback knowledge"""
        return {
            "query_classification": {
                "troubleshooting": ["failure", "problem"],
                "procedural": ["how to", "procedure"],
                "preventive": ["maintenance", "inspection"],
                "safety": ["safety", "hazard"]
            },
            "equipment_categories": {
                "equipment": ["pump", "motor", "valve"]
            },
            "technical_abbreviations": {},
            "maintie_equipment": [],
            "extracted_abbreviations": {},
            "extracted_failure_terms": [],
            "extracted_procedure_terms": []
        }

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about extracted knowledge"""
        try:
            config = self._load_existing_config()

            stats = {
                "equipment_terms": len(config.get("maintie_equipment", [])),
                "abbreviations": len(config.get("extracted_abbreviations", {})),
                "failure_terms": len(config.get("extracted_failure_terms", [])),
                "procedure_terms": len(config.get("extracted_procedure_terms", [])),
                "total_extracted": 0
            }

            stats["total_extracted"] = (
                stats["equipment_terms"] +
                stats["abbreviations"] +
                stats["failure_terms"] +
                stats["procedure_terms"]
            )

            return stats

        except Exception as e:
            logger.error(f"Error getting extraction stats: {e}")
            return {"error": str(e)}


def quick_equipment_extraction():
    """Quick script to extract equipment terms from MaintIE data"""
    print("🔍 Extracting equipment terms from MaintIE data...")

    extractor = SimpleMaintIEExtractor()

    # Load entities
    entities = extractor._load_entities()

    if not entities:
        print("❌ No processed entities found - using defaults")
        return

    # Extract equipment terms
    equipment_terms = extractor.extract_equipment_terms(entities)

    # Get most common terms
    term_counts = Counter(equipment_terms)
    common_equipment = [term for term, count in term_counts.most_common(50)]

    print(f"✅ Found {len(common_equipment)} equipment terms:")
    print(f"   Top 20: {common_equipment[:20]}")

    # Update config
    try:
        config = extractor._load_existing_config()
        config["maintie_equipment"] = common_equipment

        with open(extractor.config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ Updated {extractor.config_path} with extracted equipment terms")

    except Exception as e:
        print(f"❌ Error updating config: {e}")


def run_full_extraction():
    """Run full knowledge extraction from MaintIE data"""
    print("🔍 Running full MaintIE knowledge extraction...")

    extractor = SimpleMaintIEExtractor()
    extractor.update_domain_config()

    # Get stats
    stats = extractor.get_extraction_stats()

    print("📊 Extraction Results:")
    print(f"   Equipment terms: {stats['equipment_terms']}")
    print(f"   Abbreviations: {stats['abbreviations']}")
    print(f"   Failure terms: {stats['failure_terms']}")
    print(f"   Procedure terms: {stats['procedure_terms']}")
    print(f"   Total extracted: {stats['total_extracted']}")


if __name__ == "__main__":
    # Run quick extraction
    quick_equipment_extraction()

    # Run full extraction
    run_full_extraction()

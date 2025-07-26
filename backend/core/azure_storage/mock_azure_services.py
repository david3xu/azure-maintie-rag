"""
Mock Azure Services for Testing Dual Storage
Simulates Azure Blob, Search, and Cosmos DB for local testing
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class MockAzureBlobService:
    """Mock Azure Blob Storage Service"""
    
    def __init__(self):
        self.mock_storage_dir = Path("data/mock_azure_storage/blob")
        self.mock_storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info("MockAzureBlobService initialized")
    
    def upload_json(self, data: Dict[str, Any], blob_name: str):
        """Mock JSON upload to blob storage"""
        file_path = self.mock_storage_dir / blob_name.replace("/", "_")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Mock blob uploaded: {blob_name} -> {file_path}")


class MockAzureSearchService:
    """Mock Azure Cognitive Search Service"""
    
    def __init__(self):
        self.mock_index_dir = Path("data/mock_azure_storage/search")
        self.mock_index_dir.mkdir(parents=True, exist_ok=True)
        logger.info("MockAzureSearchService initialized")
    
    def upload_documents(self, documents: List[Dict[str, Any]]):
        """Mock document indexing"""
        index_file = self.mock_index_dir / f"search_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        with open(index_file, 'w') as f:
            for doc in documents:
                f.write(json.dumps(doc) + "\n")
        
        logger.info(f"Mock indexed {len(documents)} documents -> {index_file}")


class MockAzureCosmosService:
    """Mock Azure Cosmos DB Service"""
    
    def __init__(self):
        self.mock_cosmos_dir = Path("data/mock_azure_storage/cosmos")
        self.mock_cosmos_dir.mkdir(parents=True, exist_ok=True)
        logger.info("MockAzureCosmosService initialized")
    
    def upsert_document(self, document: Dict[str, Any]):
        """Mock document upsert"""
        doc_file = self.mock_cosmos_dir / f"{document['id']}.json"
        
        with open(doc_file, 'w') as f:
            json.dump(document, f, indent=2)
        
        logger.debug(f"Mock cosmos document saved: {document['id']}")
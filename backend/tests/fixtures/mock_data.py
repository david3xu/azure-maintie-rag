"""
Mock data for testing
Provides consistent test data across test suites
"""

from datetime import datetime
from typing import List, Dict, Any


# Sample documents for testing
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "System Maintenance Guide",
        "content": "Regular maintenance ensures optimal performance. Check components monthly.",
        "metadata": {
            "author": "Test Author",
            "created": "2024-01-01",
            "category": "maintenance"
        }
    },
    {
        "id": "doc_002",
        "title": "Troubleshooting Manual",
        "content": "Common issues include connectivity problems and performance degradation.",
        "metadata": {
            "author": "Test Author",
            "created": "2024-01-02",
            "category": "troubleshooting"
        }
    },
    {
        "id": "doc_003",
        "title": "Best Practices",
        "content": "Follow industry standards and documented procedures for best results.",
        "metadata": {
            "author": "Test Author",
            "created": "2024-01-03",
            "category": "practices"
        }
    }
]


# Sample entities for knowledge graph
SAMPLE_ENTITIES = [
    {
        "id": "entity_001",
        "name": "System",
        "type": "component",
        "properties": {
            "description": "Main system component",
            "criticality": "high"
        }
    },
    {
        "id": "entity_002",
        "name": "Performance",
        "type": "metric",
        "properties": {
            "unit": "percentage",
            "threshold": 95
        }
    },
    {
        "id": "entity_003",
        "name": "Maintenance",
        "type": "process",
        "properties": {
            "frequency": "monthly",
            "duration": "2 hours"
        }
    }
]


# Sample relationships
SAMPLE_RELATIONSHIPS = [
    {
        "id": "rel_001",
        "source": "entity_001",
        "target": "entity_002",
        "type": "affects",
        "properties": {
            "impact": "direct",
            "strength": 0.8
        }
    },
    {
        "id": "rel_002",
        "source": "entity_003",
        "target": "entity_001",
        "type": "maintains",
        "properties": {
            "required": True,
            "priority": "high"
        }
    }
]


# Sample queries
SAMPLE_QUERIES = [
    {
        "id": "query_001",
        "text": "How do I improve system performance?",
        "expected_entities": ["System", "Performance"],
        "expected_intent": "improvement"
    },
    {
        "id": "query_002",
        "text": "What maintenance tasks are required?",
        "expected_entities": ["Maintenance"],
        "expected_intent": "information"
    },
    {
        "id": "query_003",
        "text": "Troubleshoot connectivity issues",
        "expected_entities": ["System"],
        "expected_intent": "troubleshooting"
    }
]


# Sample workflow data
SAMPLE_WORKFLOW_DATA = {
    "workflow_id": "wf_test_001",
    "name": "test_workflow",
    "parameters": {
        "domain": "test",
        "input_files": ["test1.txt", "test2.txt"],
        "processing_options": {
            "extract_entities": True,
            "build_graph": True,
            "index_documents": True
        }
    },
    "expected_steps": [
        "initialization",
        "data_loading",
        "knowledge_extraction",
        "graph_construction",
        "indexing",
        "validation",
        "completion"
    ]
}


# Sample Azure service responses
SAMPLE_AZURE_RESPONSES = {
    "openai_completion": {
        "id": "completion_001",
        "choices": [{
            "message": {
                "content": "Based on the analysis, the system requires regular maintenance."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70
        }
    },
    "search_results": {
        "value": [
            {
                "@search.score": 0.95,
                "id": "doc_001",
                "content": "System maintenance guide content",
                "title": "System Maintenance Guide"
            },
            {
                "@search.score": 0.87,
                "id": "doc_002",
                "content": "Troubleshooting manual content",
                "title": "Troubleshooting Manual"
            }
        ],
        "@odata.count": 2
    },
    "storage_upload": {
        "etag": "0x8D1234567890ABC",
        "last_modified": datetime.now().isoformat(),
        "content_md5": "abc123def456",
        "blob_url": "https://storage.blob.core.windows.net/container/blob"
    }
}


def get_sample_documents(count: int = None) -> List[Dict[str, Any]]:
    """Get sample documents for testing"""
    if count:
        return SAMPLE_DOCUMENTS[:count]
    return SAMPLE_DOCUMENTS


def get_sample_entities(count: int = None) -> List[Dict[str, Any]]:
    """Get sample entities for testing"""
    if count:
        return SAMPLE_ENTITIES[:count]
    return SAMPLE_ENTITIES


def get_sample_relationships(count: int = None) -> List[Dict[str, Any]]:
    """Get sample relationships for testing"""
    if count:
        return SAMPLE_RELATIONSHIPS[:count]
    return SAMPLE_RELATIONSHIPS


def get_sample_query(query_id: str = "query_001") -> Dict[str, Any]:
    """Get a specific sample query"""
    for query in SAMPLE_QUERIES:
        if query["id"] == query_id:
            return query
    return SAMPLE_QUERIES[0]


def get_mock_azure_response(service: str) -> Dict[str, Any]:
    """Get mock Azure service response"""
    return SAMPLE_AZURE_RESPONSES.get(service, {})
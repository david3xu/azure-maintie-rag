"""
Real data fixtures for testing - NO MOCKS
Follows CODING_STANDARDS.md: Uses actual data from data/raw/ directory
"""

import pytest
from pathlib import Path
from typing import Dict, List


@pytest.fixture
def real_azure_ml_content() -> str:
    """Load REAL Azure ML documentation content from data/raw/"""
    data_file = Path("data/raw/azure-ml/azure-machine-learning-azureml-api-2.md")
    
    if not data_file.exists():
        pytest.skip(f"Real data file not found: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert len(content) > 1000, "Real data file should have substantial content"
    return content


@pytest.fixture
def real_azure_ml_chunks(real_azure_ml_content: str) -> List[str]:
    """Split real Azure ML content into chunks for processing"""
    # Split into meaningful chunks (not arbitrary splits)
    chunks = []
    lines = real_azure_ml_content.split('\n')
    current_chunk = []
    current_size = 0
    
    for line in lines:
        current_chunk.append(line)
        current_size += len(line)
        
        # Create chunks around 1500 characters for realistic processing
        if current_size > 1500 and line.strip() == "":
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)
            current_chunk = []
            current_size = 0
    
    # Add remaining content
    if current_chunk:
        chunk_text = '\n'.join(current_chunk).strip()
        if chunk_text:
            chunks.append(chunk_text)
    
    assert len(chunks) > 0, "Should have created chunks from real data"
    return chunks


@pytest.fixture
def real_domain_queries() -> List[str]:
    """Real domain-specific queries based on actual data content"""
    return [
        "How do I deploy a machine learning model in Azure?",
        "What is Azure Machine Learning prompt flow?",
        "How to create data assets in Azure ML?",
        "What are Azure ML pipelines and how do I build them?",
        "How to manage the ML lifecycle with MLOps in Azure?",
        "How to set up secure workspace in Azure Machine Learning?",
        "What is responsible AI in Azure Machine Learning?",
        "How to use Apache Spark in Azure Machine Learning?",
        "How to track and monitor training runs in Azure ML?",
        "What are online endpoints for real-time scoring?"
    ]


@pytest.fixture 
def real_expected_entities() -> List[str]:
    """Expected entities that should be found in real Azure ML data"""
    return [
        "Azure Machine Learning",
        "MLOps",
        "model deployment", 
        "prompt flow",
        "data assets",
        "pipelines",
        "online endpoints",
        "Apache Spark",
        "training runs",
        "workspace",
        "responsible AI",
        "real-time scoring"
    ]


@pytest.fixture
def real_performance_thresholds() -> Dict[str, float]:
    """Real performance thresholds based on actual requirements"""
    return {
        "knowledge_extraction_per_chunk": 5.0,  # seconds
        "search_query": 3.0,  # seconds  
        "storage_upload_per_mb": 2.0,  # seconds
        "health_check": 10.0,  # seconds
        "workflow_step": 2.0,  # seconds per step
        "universal_query_processing": 30.0,  # seconds total
        "service_initialization": 15.0  # seconds
    }


@pytest.fixture
def azure_credentials_required():
    """Fixture that ensures Azure credentials are configured"""
    from config.settings import settings
    
    missing_credentials = []
    
    if not settings.azure_openai_endpoint:
        missing_credentials.append("AZURE_OPENAI_ENDPOINT")
    if not settings.azure_search_endpoint:
        missing_credentials.append("AZURE_SEARCH_ENDPOINT") 
    if not settings.azure_storage_account:
        missing_credentials.append("AZURE_STORAGE_ACCOUNT")
    if not settings.azure_cosmos_endpoint:
        missing_credentials.append("AZURE_COSMOS_ENDPOINT")
    
    if missing_credentials:
        pytest.skip(
            f"Azure credentials required for real testing: {', '.join(missing_credentials)}\n"
            f"Set environment variables or configure in .env file"
        )
    
    return settings


class RealDataValidator:
    """Validator for ensuring test data is real (not fake/mock)"""
    
    @staticmethod
    def validate_no_mock_data(data: any, context: str = ""):
        """Validate that data doesn't contain mock/fake indicators"""
        data_str = str(data).lower()
        
        forbidden_terms = [
            "mock", "fake", "placeholder", "dummy", "test_", 
            "example_", "sample_", "demo_", "stub"
        ]
        
        for term in forbidden_terms:
            assert term not in data_str, f"Found forbidden term '{term}' in {context}: {data_str[:200]}"
    
    @staticmethod
    def validate_real_content(content: str, min_length: int = 100):
        """Validate content appears to be real (not generated/fake)"""
        assert len(content) >= min_length, f"Content too short to be real: {len(content)} chars"
        assert not content.startswith("Mock"), "Content appears to be mock data"
        assert not content.startswith("Test"), "Content appears to be test data"
        
        # Real content should have some variety in sentence structure
        sentences = content.split('. ')
        if len(sentences) > 3:
            sentence_lengths = [len(s) for s in sentences[:10]]
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            assert avg_length > 20, "Content appears too uniform to be real"
    
    @staticmethod
    def validate_real_performance(duration: float, operation: str, threshold: float):
        """Validate performance timing appears real (not instant fake responses)"""
        assert duration > 0.01, f"{operation} completed too fast to be real: {duration:.4f}s"
        assert duration < threshold, f"{operation} took too long: {duration:.2f}s > {threshold}s"
        
        # Real operations should have some variance (not exactly the same time)
        assert duration != 1.0, f"{operation} took exactly 1.0s - suspicious"
        assert duration != 0.5, f"{operation} took exactly 0.5s - suspicious"


@pytest.fixture
def real_data_validator():
    """Fixture providing real data validation utilities"""
    return RealDataValidator()
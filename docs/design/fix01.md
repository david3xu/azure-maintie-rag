# Azure Knowledge Extraction Enterprise Architecture Implementation

**Based on Real Codebase Analysis**
**Target Files**: `backend/core/azure_openai/knowledge_extractor.py`, `backend/core/azure_openai/extraction_client.py`
**Azure Services**: Text Analytics, Application Insights, ML Endpoints, Cognitive Search

---

## 🏗️ Enterprise Architecture Overview

### Core Service Components
```
AzureKnowledgeOrchestrator
├── AzureTextAnalyticsService (Pre-processing)
├── AzureOpenAIExtractionService (LLM Processing)
├── AzureCognitiveSearchValidator (Quality Validation)
├── AzureMLQualityAssessment (Confidence Scoring)
└── AzureMonitoringService (Real-time Telemetry)
```

### Data Flow Architecture
```
Raw Text → Azure Text Analytics → Azure OpenAI → Quality Validation → Azure Cosmos DB
    ↓              ↓                   ↓              ↓                ↓
Pre-process    Entity Extraction   Confidence      Graph Storage   Monitoring
Optimization   + Relations         Calibration     + Indexing      Dashboard
```

---

## 📋 Implementation Instructions

### **1. Azure Text Analytics Integration Service**

**File**: `backend/core/azure_openai/azure_text_analytics_service.py` (New)

```python
"""
Azure Text Analytics Service for Enterprise Knowledge Extraction
Pre-processing service to enhance extraction accuracy
"""

from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from typing import Dict, List, Any, Optional
import logging
from ...config.settings import azure_settings

logger = logging.getLogger(__name__)

class AzureTextAnalyticsService:
    """Enterprise text pre-processing with Azure Cognitive Services"""

    def __init__(self):
        self.client = TextAnalyticsClient(
            endpoint=azure_settings.azure_text_analytics_endpoint,
            credential=AzureKeyCredential(azure_settings.azure_text_analytics_key)
        )
        self.supported_languages = ["en", "es", "fr", "de"]

    async def preprocess_for_extraction(self, texts: List[str]) -> Dict[str, Any]:
        """
        Pre-process texts using Azure Text Analytics for enhanced extraction
        Returns enhanced context for knowledge extraction
        """
        # Detect language and entities
        language_results = await self._detect_language_batch(texts)
        entity_results = await self._recognize_entities_batch(texts)
        key_phrase_results = await self._extract_key_phrases_batch(texts)

        return {
            "enhanced_texts": texts,
            "language_metadata": language_results,
            "pre_identified_entities": entity_results,
            "key_phrases": key_phrase_results,
            "processing_confidence": self._calculate_preprocessing_confidence(
                language_results, entity_results
            )
        }

    async def _detect_language_batch(self, texts: List[str]) -> List[Dict]:
        """Batch language detection with confidence scoring"""
        try:
            response = self.client.detect_language(documents=texts)
            return [
                {
                    "language": doc.primary_language.iso6391_name,
                    "confidence": doc.primary_language.confidence_score,
                    "supported": doc.primary_language.iso6391_name in self.supported_languages
                }
                for doc in response if not doc.is_error
            ]
        except Exception as e:
            logger.error(f"Azure Text Analytics language detection failed: {e}")
            return [{"language": "en", "confidence": 0.5, "supported": True}] * len(texts)
```

**Configuration Update**: `backend/config/settings.py`
```python
# Add Azure Text Analytics settings
azure_text_analytics_endpoint: str = Field(default="", env="AZURE_TEXT_ANALYTICS_ENDPOINT")
azure_text_analytics_key: str = Field(default="", env="AZURE_TEXT_ANALYTICS_KEY")
```

### **2. Enhanced Knowledge Extractor with Azure ML Quality Assessment**

**File**: `backend/core/azure_openai/knowledge_extractor.py` (Modify existing)

**Add Azure ML Quality Service Integration**:
```python
# Import additions at top of file
from .azure_text_analytics_service import AzureTextAnalyticsService
from .azure_ml_quality_service import AzureMLQualityAssessment
from .azure_monitoring_service import AzureKnowledgeMonitor

class AzureOpenAIKnowledgeExtractor:
    """Enhanced with Azure ML quality assessment and monitoring"""

    def __init__(self, domain_name: str = "general"):
        # Existing initialization...

        # Add Azure services
        self.text_analytics = AzureTextAnalyticsService()
        self.quality_assessor = AzureMLQualityAssessment(domain_name)
        self.monitor = AzureKnowledgeMonitor()

        # Enterprise extraction configuration
        self.extraction_config = self._load_extraction_config()

    def _load_extraction_config(self) -> Dict[str, Any]:
        """Load domain-specific extraction configuration"""
        return {
            "quality_tier": azure_settings.extraction_quality_tier,
            "confidence_threshold": azure_settings.extraction_confidence_threshold,
            "max_entities_per_document": azure_settings.max_entities_per_document,
            "batch_size": azure_settings.extraction_batch_size,
            "enable_preprocessing": azure_settings.enable_text_analytics_preprocessing
        }
```

**Replace Static Quality Assessment**:
```python
# Replace existing _assess_extraction_quality method
async def _assess_extraction_quality(self) -> Dict[str, Any]:
    """Enterprise quality assessment using Azure ML models"""

    # Prepare extraction context for ML assessment
    extraction_context = {
        "domain": self.domain_name,
        "entity_count": len(self.entities),
        "relation_count": len(self.relations),
        "entity_types": list(self.discovered_entity_types),
        "relation_types": list(self.discovered_relation_types),
        "documents_processed": len(self.documents),
        "extraction_config": self.extraction_config
    }

    # Use Azure ML for quality assessment
    quality_results = await self.quality_assessor.assess_extraction_quality(
        extraction_context,
        self.entities,
        self.relations
    )

    # Track quality metrics in Azure Monitor
    await self.monitor.track_extraction_quality(quality_results)

    return quality_results
```

### **3. Azure ML Quality Assessment Service**

**File**: `backend/core/azure_openai/azure_ml_quality_service.py` (New)

```python
"""
Azure ML Quality Assessment Service
Enterprise-grade quality scoring using Azure ML endpoints
"""

from azure.ml.models import MLModel
from azure.core.credentials import DefaultAzureCredential
import asyncio
import logging
from typing import Dict, List, Any
from ...config.settings import azure_settings

logger = logging.getLogger(__name__)

class AzureMLQualityAssessment:
    """Azure ML-powered quality assessment for knowledge extraction"""

    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.credential = DefaultAzureCredential()
        self.ml_client = self._initialize_ml_client()

        # Quality assessment models
        self.confidence_model_endpoint = azure_settings.azure_ml_confidence_endpoint
        self.completeness_model_endpoint = azure_settings.azure_ml_completeness_endpoint

    def _initialize_ml_client(self):
        """Initialize Azure ML client with managed identity"""
        from azure.ai.ml import MLClient
        return MLClient(
            credential=self.credential,
            subscription_id=azure_settings.azure_subscription_id,
            resource_group_name=azure_settings.azure_resource_group,
            workspace_name=azure_settings.azure_ml_workspace
        )

    async def assess_extraction_quality(
        self,
        extraction_context: Dict[str, Any],
        entities: Dict[str, Any],
        relations: List[Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive quality assessment using Azure ML models
        Returns enterprise-grade quality metrics
        """

        # Parallel assessment tasks
        confidence_task = self._assess_confidence_distribution(entities, relations)
        completeness_task = self._assess_domain_completeness(extraction_context)
        consistency_task = self._assess_semantic_consistency(entities, relations)

        # Execute assessments concurrently
        confidence_results, completeness_results, consistency_results = await asyncio.gather(
            confidence_task, completeness_task, consistency_task
        )

        # Aggregate enterprise quality score
        enterprise_score = self._calculate_enterprise_quality_score(
            confidence_results, completeness_results, consistency_results
        )

        return {
            "enterprise_quality_score": enterprise_score,
            "confidence_assessment": confidence_results,
            "completeness_assessment": completeness_results,
            "consistency_assessment": consistency_results,
            "quality_tier": self._determine_quality_tier(enterprise_score),
            "recommendations": self._generate_improvement_recommendations(
                confidence_results, completeness_results, consistency_results
            ),
            "azure_ml_metadata": {
                "confidence_model": self.confidence_model_endpoint,
                "completeness_model": self.completeness_model_endpoint,
                "assessment_timestamp": datetime.now().isoformat()
            }
        }

    async def _assess_confidence_distribution(
        self,
        entities: Dict[str, Any],
        relations: List[Any]
    ) -> Dict[str, Any]:
        """Azure ML-based confidence distribution analysis"""

        # Prepare features for ML model
        confidence_features = {
            "entity_confidence_distribution": [e.confidence for e in entities.values()],
            "relation_confidence_distribution": [r.confidence for r in relations],
            "confidence_variance": self._calculate_confidence_variance(entities, relations),
            "low_confidence_ratio": self._calculate_low_confidence_ratio(entities, relations)
        }

        # Call Azure ML confidence assessment endpoint
        try:
            confidence_assessment = await self._call_ml_endpoint(
                self.confidence_model_endpoint,
                confidence_features
            )
            return confidence_assessment
        except Exception as e:
            logger.error(f"Azure ML confidence assessment failed: {e}")
            return {"confidence_score": 0.5, "assessment_quality": "degraded"}
```

### **4. Azure Application Insights Monitoring Service**

**File**: `backend/core/azure_openai/azure_monitoring_service.py` (New)

```python
"""
Azure Application Insights Integration for Knowledge Extraction Monitoring
Real-time telemetry and performance tracking
"""

from azure.monitor.opentelemetry import configure_azure_monitor
from azure.applicationinsights import TelemetryClient
import logging
from typing import Dict, Any
from ...config.settings import azure_settings

logger = logging.getLogger(__name__)

class AzureKnowledgeMonitor:
    """Enterprise monitoring for knowledge extraction pipeline"""

    def __init__(self):
        self.telemetry_client = TelemetryClient(
            azure_settings.azure_application_insights_connection_string
        )

        # Configure OpenTelemetry for Azure Monitor
        configure_azure_monitor(
            connection_string=azure_settings.azure_application_insights_connection_string
        )

        self.custom_metrics = {
            "extraction_quality_score": "knowledge_extraction.quality.score",
            "entities_extracted": "knowledge_extraction.entities.count",
            "relations_extracted": "knowledge_extraction.relations.count",
            "processing_duration": "knowledge_extraction.processing.duration_ms",
            "azure_openai_tokens_used": "knowledge_extraction.azure_openai.tokens",
            "confidence_distribution": "knowledge_extraction.confidence.distribution"
        }

    async def track_extraction_quality(self, quality_results: Dict[str, Any]) -> None:
        """Track quality metrics in Azure Application Insights"""

        # Track enterprise quality score
        self.telemetry_client.track_metric(
            self.custom_metrics["extraction_quality_score"],
            quality_results.get("enterprise_quality_score", 0.0),
            properties={
                "domain": quality_results.get("domain", "unknown"),
                "quality_tier": quality_results.get("quality_tier", "unknown")
            }
        )

        # Track extraction counts
        self.telemetry_client.track_metric(
            self.custom_metrics["entities_extracted"],
            quality_results.get("entity_count", 0)
        )

        self.telemetry_client.track_metric(
            self.custom_metrics["relations_extracted"],
            quality_results.get("relation_count", 0)
        )

        # Track quality recommendations as custom events
        if "recommendations" in quality_results:
            self.telemetry_client.track_event(
                "knowledge_extraction_recommendations",
                properties={
                    "recommendations": quality_results["recommendations"],
                    "quality_score": quality_results.get("enterprise_quality_score", 0.0)
                }
            )

        # Flush telemetry
        self.telemetry_client.flush()

    async def track_azure_openai_usage(
        self,
        tokens_used: int,
        api_calls: int,
        processing_time_ms: float
    ) -> None:
        """Track Azure OpenAI usage and costs"""

        self.telemetry_client.track_metric(
            self.custom_metrics["azure_openai_tokens_used"],
            tokens_used,
            properties={
                "api_calls": str(api_calls),
                "avg_tokens_per_call": str(tokens_used / api_calls if api_calls > 0 else 0)
            }
        )

        self.telemetry_client.track_metric(
            self.custom_metrics["processing_duration"],
            processing_time_ms
        )
```

### **5. Cost Optimization and Rate Limiting Service**

**File**: `backend/core/azure_openai/azure_rate_limiter.py` (New)

```python
"""
Azure OpenAI Rate Limiting and Cost Optimization Service
Enterprise-grade quota management and cost control
"""

import asyncio
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
from ...config.settings import azure_settings

logger = logging.getLogger(__name__)

@dataclass
class AzureOpenAIQuota:
    """Azure OpenAI quota configuration"""
    max_tokens_per_minute: int
    max_requests_per_minute: int
    cost_threshold_per_hour: float
    priority_tier: str  # "enterprise", "standard", "cost_optimized"

class AzureOpenAIRateLimiter:
    """Enterprise rate limiting and cost optimization"""

    def __init__(self):
        self.quota_config = self._load_quota_config()
        self.usage_tracker = {
            "tokens_used_this_minute": 0,
            "requests_this_minute": 0,
            "cost_this_hour": 0.0,
            "last_reset_time": time.time()
        }

    def _load_quota_config(self) -> AzureOpenAIQuota:
        """Load quota configuration from Azure settings"""
        return AzureOpenAIQuota(
            max_tokens_per_minute=azure_settings.azure_openai_max_tokens_per_minute,
            max_requests_per_minute=azure_settings.azure_openai_max_requests_per_minute,
            cost_threshold_per_hour=azure_settings.azure_openai_cost_threshold_per_hour,
            priority_tier=azure_settings.azure_openai_priority_tier
        )

    async def execute_with_rate_limiting(
        self,
        extraction_function: callable,
        estimated_tokens: int,
        priority: str = "standard"
    ) -> Any:
        """
        Execute extraction with enterprise rate limiting
        Includes exponential backoff and cost controls
        """

        # Check quota availability
        if not await self._check_quota_availability(estimated_tokens):
            await self._wait_for_quota_reset()

        # Execute with retry logic
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Track usage before execution
                start_time = time.time()

                # Execute extraction
                result = await extraction_function()

                # Track actual usage
                execution_time = time.time() - start_time
                actual_tokens = self._estimate_tokens_from_result(result)

                await self._update_usage_tracking(actual_tokens, execution_time)

                return result

            except Exception as e:
                if attempt == max_retries - 1:
                    raise e

                # Exponential backoff
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Azure OpenAI call failed, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)
```

### **6. Configuration Updates**

**File**: `backend/config/settings.py` (Add Azure ML and monitoring settings)

```python
# Azure ML Quality Assessment Settings
azure_ml_confidence_endpoint: str = Field(default="", env="AZURE_ML_CONFIDENCE_ENDPOINT")
azure_ml_completeness_endpoint: str = Field(default="", env="AZURE_ML_COMPLETENESS_ENDPOINT")

# Azure Text Analytics Settings
azure_text_analytics_endpoint: str = Field(default="", env="AZURE_TEXT_ANALYTICS_ENDPOINT")
azure_text_analytics_key: str = Field(default="", env="AZURE_TEXT_ANALYTICS_KEY")

# Knowledge Extraction Configuration
extraction_quality_tier: str = Field(default="standard", env="EXTRACTION_QUALITY_TIER")
extraction_confidence_threshold: float = Field(default=0.7, env="EXTRACTION_CONFIDENCE_THRESHOLD")
max_entities_per_document: int = Field(default=100, env="MAX_ENTITIES_PER_DOCUMENT")
extraction_batch_size: int = Field(default=10, env="EXTRACTION_BATCH_SIZE")
enable_text_analytics_preprocessing: bool = Field(default=True, env="ENABLE_TEXT_ANALYTICS_PREPROCESSING")

# Azure OpenAI Rate Limiting
azure_openai_max_tokens_per_minute: int = Field(default=40000, env="AZURE_OPENAI_MAX_TOKENS_PER_MINUTE")
azure_openai_max_requests_per_minute: int = Field(default=60, env="AZURE_OPENAI_MAX_REQUESTS_PER_MINUTE")
azure_openai_cost_threshold_per_hour: float = Field(default=50.0, env="AZURE_OPENAI_COST_THRESHOLD_PER_HOUR")
azure_openai_priority_tier: str = Field(default="standard", env="AZURE_OPENAI_PRIORITY_TIER")
```

---

## 🚀 Deployment and Integration Steps

### **1. Azure Services Provisioning**

**Infrastructure Updates** (Add to bicep templates):
```bicep
// Azure Text Analytics
resource textAnalytics 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: '${resourcePrefix}-text-analytics'
  location: location
  sku: {
    name: 'S'
  }
  kind: 'TextAnalytics'
}

// Azure ML Endpoints for Quality Assessment
resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2023-04-01' = {
  // Reference existing ML workspace
  // Deploy quality assessment models
}
```

### **2. Monitoring Dashboard Setup**

**Azure Application Insights Queries**:
```kusto
// Knowledge extraction quality trending
customMetrics
| where name == "knowledge_extraction.quality.score"
| summarize avg(value) by bin(timestamp, 1h)
| render timechart

// Cost tracking
customMetrics
| where name == "knowledge_extraction.azure_openai.tokens"
| summarize sum(value) by bin(timestamp, 1d)
| extend daily_cost = value * 0.002  // Approximate token cost
```

### **3. Testing and Validation**

**Enterprise Testing Script**:
```python
# File: backend/scripts/test_enterprise_knowledge_extraction.py
async def test_enterprise_extraction():
    """Test enhanced knowledge extraction with Azure services"""

    extractor = AzureOpenAIKnowledgeExtractor("enterprise_test")

    # Test with sample documents
    test_texts = ["Sample enterprise document content..."]

    results = await extractor.extract_knowledge_from_texts(test_texts)

    # Validate enterprise quality metrics
    assert results["success"]
    assert "enterprise_quality_score" in results["quality_assessment"]
    assert results["quality_assessment"]["enterprise_quality_score"] > 0.7
```

This enterprise architecture provides Azure-native knowledge extraction with quality assurance, cost optimization, and real-time monitoring capabilities.
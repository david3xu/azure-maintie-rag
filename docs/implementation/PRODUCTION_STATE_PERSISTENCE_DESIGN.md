# Production State Persistence Design

**Date**: August 3, 2025  
**Status**: 🚧 **DESIGN PHASE** - Production-grade state persistence for pydantic-graph workflows

## Overview

This document defines the production-grade state persistence strategy for Azure Universal RAG's graph-based workflow system, replacing the current orchestrator-based approach with fault-tolerant, scalable state management.

## Current Performance Baseline

**CRITICAL PERFORMANCE ISSUE IDENTIFIED**:
- **Current orchestrator initialization**: 2.8s
- **Total workflow time**: 3.2s  
- **SLA Status**: ❌ **FAILS** sub-3-second requirement by 200ms

## State Persistence Requirements

### **Primary Use Cases**
1. **Long-Running Azure Operations**: Azure ML training (5-15 minutes), large corpus analysis (hours)
2. **Fault Recovery**: Resume workflows after Azure service timeouts/failures
3. **Multi-Environment Support**: Dev/staging/prod with different persistence backends
4. **State Auditing**: Compliance tracking for workflow state changes
5. **Performance Optimization**: Cache workflow states for repeated operations

### **Production Requirements**
- **Encryption**: All state data encrypted at rest using Azure Key Vault
- **Performance**: State read/write operations under 50ms  
- **Scalability**: Support 1000+ concurrent workflows
- **Reliability**: 99.9% availability with automatic failover
- **Compliance**: GDPR-compliant data retention policies

## Architecture Design

### **Multi-Tier Persistence Strategy**

```python
# Production persistence hierarchy
class ProductionStatePersistence:
    """
    Multi-tier state persistence with performance optimization
    """
    def __init__(self):
        # Tier 1: In-memory cache (Redis) - <5ms access
        self.memory_cache = AzureRedisCache()
        
        # Tier 2: Fast persistent storage (PostgreSQL) - <50ms access  
        self.persistent_db = AzurePostgreSQLFlexible()
        
        # Tier 3: Long-term archival (Azure Storage) - <500ms access
        self.archive_storage = AzureStorageAccount()
        
        # Encryption service
        self.encryption = AzureKeyVaultEncryption()
```

### **State Storage Tiers**

#### **Tier 1: Redis Cache (Hot State)**
- **Purpose**: Active workflow states requiring <5ms access
- **Technology**: Azure Cache for Redis (Premium tier)
- **Data**: Current graph node states, intermediate results
- **TTL**: 24 hours for active workflows
- **Size**: Up to 1GB per workflow state

#### **Tier 2: PostgreSQL (Warm State)**  
- **Purpose**: Persistent workflow history and recovery data
- **Technology**: Azure Database for PostgreSQL Flexible Server
- **Data**: Complete workflow execution logs, final results
- **Retention**: 90 days for compliance
- **Features**: Point-in-time recovery, automated backups

#### **Tier 3: Azure Storage (Cold State)**
- **Purpose**: Long-term archival and compliance data
- **Technology**: Azure Storage Account (Cool tier)
- **Data**: Completed workflow states, audit logs
- **Retention**: 7 years for regulatory compliance
- **Features**: Immutable storage, lifecycle policies

## Implementation Architecture

### **State Models**

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

class WorkflowStage(str, Enum):
    DOMAIN_ANALYSIS = "domain_analysis"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction" 
    SEARCH_ORCHESTRATION = "search_orchestration"
    RESULT_SYNTHESIS = "result_synthesis"

class WorkflowState(BaseModel):
    """Complete workflow state for Config-Extraction graph"""
    workflow_id: str = Field(description="Unique workflow identifier")
    user_id: str = Field(description="User identifier for access control")
    current_stage: WorkflowStage = Field(description="Current workflow stage")
    
    # Domain analysis results
    raw_data: Optional[str] = Field(default=None, description="Input corpus data")
    domain_config: Optional[Dict[str, Any]] = Field(default=None, description="Generated domain configuration")
    
    # Knowledge extraction results  
    extracted_entities: Optional[List[Dict]] = Field(default=None, description="Extracted entities")
    extracted_relationships: Optional[List[Dict]] = Field(default=None, description="Extracted relationships")
    
    # Search orchestration results
    search_results: Optional[Dict[str, Any]] = Field(default=None, description="Tri-modal search results")
    final_response: Optional[str] = Field(default=None, description="Final synthesized response")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    azure_services_used: List[str] = Field(default_factory=list, description="Azure services utilized")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance tracking")
    
class EncryptedState(BaseModel):
    """Encrypted state wrapper for secure storage"""
    state_id: str
    encrypted_data: bytes
    encryption_key_id: str  # Azure Key Vault key reference
    checksum: str
    created_at: datetime
```

### **Production State Manager**

```python
import asyncio
import json
import hashlib
from typing import Optional
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from redis.asyncio import Redis
import asyncpg

class ProductionStateManager:
    """Production-grade state persistence with encryption and multi-tier storage"""
    
    def __init__(self):
        self.credential = DefaultAzureCredential()
        self.key_vault_client = SecretClient(
            vault_url="https://kv-maintie-rag-prod.vault.azure.net/",
            credential=self.credential
        )
        
        # Initialize storage tiers
        self.redis_client: Optional[Redis] = None
        self.postgres_pool: Optional[asyncpg.Pool] = None
        
    async def initialize(self):
        """Initialize all storage connections"""
        # Redis connection
        redis_connection_string = await self._get_secret("redis-connection-string")
        self.redis_client = Redis.from_url(redis_connection_string)
        
        # PostgreSQL connection pool
        postgres_connection_string = await self._get_secret("postgres-connection-string")
        self.postgres_pool = await asyncpg.create_pool(postgres_connection_string)
        
    async def save_state(self, workflow_id: str, state: WorkflowState) -> None:
        """Save workflow state with encryption across tiers"""
        try:
            # Encrypt state data
            encrypted_state = await self._encrypt_state(state)
            
            # Tier 1: Save to Redis (hot cache)
            await self.redis_client.setex(
                f"workflow:{workflow_id}",
                86400,  # 24 hour TTL
                encrypted_state.model_dump_json()
            )
            
            # Tier 2: Save to PostgreSQL (persistent)
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO workflow_states (
                        workflow_id, encrypted_data, encryption_key_id, 
                        stage, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (workflow_id) 
                    DO UPDATE SET 
                        encrypted_data = $2,
                        stage = $4,
                        updated_at = $6
                """, workflow_id, encrypted_state.encrypted_data, encrypted_state.encryption_key_id,
                     state.current_stage, state.created_at, state.updated_at)
                     
        except Exception as e:
            # Log error and raise for retry logic
            logger.error(f"State save failed for workflow {workflow_id}: {e}")
            raise
            
    async def load_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Load workflow state with automatic tier fallback"""
        try:
            # Try Tier 1: Redis first (fastest)
            cached_state = await self.redis_client.get(f"workflow:{workflow_id}")
            if cached_state:
                encrypted_state = EncryptedState.model_validate_json(cached_state)
                return await self._decrypt_state(encrypted_state)
            
            # Fallback to Tier 2: PostgreSQL
            async with self.postgres_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT encrypted_data, encryption_key_id, created_at 
                    FROM workflow_states 
                    WHERE workflow_id = $1
                """, workflow_id)
                
                if row:
                    encrypted_state = EncryptedState(
                        state_id=workflow_id,
                        encrypted_data=row['encrypted_data'],
                        encryption_key_id=row['encryption_key_id'],
                        checksum="",  # Calculate if needed
                        created_at=row['created_at']
                    )
                    
                    # Restore to Redis cache
                    decrypted_state = await self._decrypt_state(encrypted_state)
                    await self.save_state(workflow_id, decrypted_state)
                    return decrypted_state
                    
            return None
            
        except Exception as e:
            logger.error(f"State load failed for workflow {workflow_id}: {e}")
            return None
    
    async def _encrypt_state(self, state: WorkflowState) -> EncryptedState:
        """Encrypt state using Azure Key Vault"""
        # Get encryption key from Key Vault
        encryption_key = await self._get_secret("state-encryption-key")
        
        # Serialize and encrypt state
        state_json = state.model_dump_json()
        encrypted_data = self._encrypt_data(state_json.encode(), encryption_key)
        
        return EncryptedState(
            state_id=state.workflow_id,
            encrypted_data=encrypted_data,
            encryption_key_id="state-encryption-key",
            checksum=hashlib.sha256(encrypted_data).hexdigest(),
            created_at=datetime.utcnow()
        )
    
    async def _decrypt_state(self, encrypted_state: EncryptedState) -> WorkflowState:
        """Decrypt state using Azure Key Vault"""
        # Get decryption key
        decryption_key = await self._get_secret(encrypted_state.encryption_key_id)
        
        # Decrypt and deserialize
        decrypted_data = self._decrypt_data(encrypted_state.encrypted_data, decryption_key)
        state_dict = json.loads(decrypted_data.decode())
        
        return WorkflowState.model_validate(state_dict)
    
    async def _get_secret(self, secret_name: str) -> str:
        """Retrieve secret from Azure Key Vault"""
        secret = await self.key_vault_client.get_secret(secret_name)
        return secret.value
    
    def _encrypt_data(self, data: bytes, key: str) -> bytes:
        """Encrypt data using AES-256"""
        # Implementation using cryptography library
        pass
    
    def _decrypt_data(self, encrypted_data: bytes, key: str) -> bytes:
        """Decrypt data using AES-256"""
        # Implementation using cryptography library  
        pass
```

## Graph Integration

### **pydantic-graph State Integration**

```python
from pydantic_graph import Graph, BaseNode, GraphRunContext
from .persistence import ProductionStateManager, WorkflowState

class ConfigExtractionGraph:
    """Production graph with state persistence"""
    
    def __init__(self):
        self.state_manager = ProductionStateManager()
        
    async def run_with_persistence(
        self, 
        workflow_id: str, 
        initial_data: str
    ) -> WorkflowState:
        """Execute graph with automatic state persistence"""
        
        # Initialize or restore state
        state = await self.state_manager.load_state(workflow_id)
        if not state:
            state = WorkflowState(
                workflow_id=workflow_id,
                raw_data=initial_data,
                current_stage=WorkflowStage.DOMAIN_ANALYSIS
            )
        
        # Execute graph nodes with state checkpoints
        try:
            async with self.graph.iter(
                start_node=self._get_resume_node(state),
                state=state,
                persistence=self.state_manager
            ) as run:
                async for node in run:
                    # Save state after each node completion
                    await self.state_manager.save_state(workflow_id, run.state)
                    
                    # Performance tracking
                    run.state.performance_metrics[node.__class__.__name__] = run.node_execution_time
                    
                    # SLA monitoring
                    if run.total_execution_time > 3.0:
                        logger.warning(f"SLA violation: {run.total_execution_time:.2f}s")
                        
                return run.result
                
        except Exception as e:
            # Save error state for debugging
            state.error_details = {"error": str(e), "stage": state.current_stage}
            await self.state_manager.save_state(workflow_id, state)
            raise
```

## Infrastructure Requirements

### **Azure Resources**

```bicep
// Azure infrastructure for state persistence

resource redisCache 'Microsoft.Cache/Redis@2023-04-01' = {
  name: 'redis-maintie-rag-prod'
  location: location
  properties: {
    sku: {
      name: 'Premium'
      family: 'P'
      capacity: 1
    }
    redisConfiguration: {
      'maxmemory-policy': 'allkeys-lru'
    }
    enableNonSslPort: false
  }
}

resource postgresqlServer 'Microsoft.DBforPostgreSQL/flexibleServers@2023-03-01-preview' = {
  name: 'postgres-maintie-rag-prod'
  location: location
  sku: {
    name: 'Standard_D2ds_v4'
    tier: 'GeneralPurpose'
  }
  properties: {
    administratorLogin: 'ragadmin'
    storage: {
      storageSizeGB: 128
    }
    backup: {
      backupRetentionDays: 35
      geoRedundantBackup: 'Enabled'
    }
  }
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-02-01' = {
  name: 'kv-maintie-rag-prod'
  location: location
  properties: {
    tenantId: tenant().tenantId
    sku: {
      family: 'A'
      name: 'premium'  // Hardware security modules
    }
    enabledForDeployment: true
    enabledForDiskEncryption: true
  }
}
```

### **Database Schema**

```sql
-- PostgreSQL schema for workflow state persistence
CREATE TABLE workflow_states (
    workflow_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    encrypted_data BYTEA NOT NULL,
    encryption_key_id VARCHAR(255) NOT NULL,
    stage VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    archived_at TIMESTAMP WITH TIME ZONE,
    
    -- Performance tracking
    execution_time_ms INTEGER,
    azure_services_used TEXT[],
    
    -- Indexing for fast lookups
    INDEX idx_workflow_user (user_id),
    INDEX idx_workflow_stage (stage),
    INDEX idx_workflow_created (created_at)
);

-- Audit table for compliance
CREATE TABLE workflow_audit_log (
    id SERIAL PRIMARY KEY,
    workflow_id VARCHAR(255) NOT NULL,
    action VARCHAR(50) NOT NULL, -- 'created', 'updated', 'completed', 'failed'
    stage VARCHAR(50),
    user_id VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    details JSONB
);

-- Archival table for long-term storage
CREATE TABLE archived_workflows (
    workflow_id VARCHAR(255) PRIMARY KEY,
    archived_data BYTEA NOT NULL,
    archived_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    retention_until TIMESTAMP WITH TIME ZONE NOT NULL
);
```

## Cost Analysis

### **Monthly Cost Estimates (Production)**

| Resource | Configuration | Monthly Cost | Purpose |
|----------|---------------|--------------|---------|
| **Azure Redis Cache** | Premium P1 (6GB) | $250 | Hot state cache |
| **PostgreSQL Flexible** | Standard_D2ds_v4 | $180 | Persistent storage |
| **Azure Storage** | Cool tier (1TB) | $15 | Long-term archival |
| **Key Vault** | Premium HSM | $80 | Encryption keys |
| **Network/Bandwidth** | Data transfer | $20 | Inter-service communication |
| **Total** | | **$545/month** | Complete persistence infrastructure |

### **ROI Analysis**
- **Infrastructure Cost**: $545/month
- **Performance Improvement**: 200ms reduction (from current 3.2s failure)
- **Reliability Improvement**: 90% fewer workflow restarts
- **Development Time Saved**: 50% reduction in debugging effort
- **Break-even**: 2-3 months based on development productivity gains

## Migration Strategy

### **Phase 1: Infrastructure Setup (Week 1)**
1. Deploy Azure resources (Redis, PostgreSQL, Key Vault)
2. Configure networking and security
3. Set up database schema and encryption keys
4. Implement basic state manager without graph integration

### **Phase 2: Graph Integration (Week 2)**  
1. Integrate state manager with pydantic-graph
2. Implement automatic checkpointing
3. Add performance monitoring and SLA tracking
4. Create fallback mechanisms for state load failures

### **Phase 3: Production Deployment (Week 3)**
1. Parallel deployment alongside existing orchestrators
2. A/B testing with traffic splitting (10% graph, 90% orchestrators)
3. Performance validation and SLA monitoring
4. Gradual traffic migration based on success metrics

### **Phase 4: Optimization (Week 4)**
1. Performance tuning based on production metrics
2. Cost optimization and resource scaling
3. Complete migration to graph-based workflows
4. Decommission legacy orchestrator infrastructure

## Success Metrics

### **Performance KPIs**
- **State Persistence Latency**: <50ms for read/write operations
- **Workflow Recovery Time**: <500ms from failure to resume
- **Total Workflow Time**: <2.5s (500ms improvement from baseline)
- **Cache Hit Rate**: >95% for active workflows

### **Reliability KPIs**  
- **System Availability**: 99.9% uptime
- **State Persistence Success**: 99.99% success rate
- **Automatic Recovery**: 90% of failures auto-recover without human intervention
- **Data Integrity**: 100% state consistency validation

### **Cost KPIs**
- **Infrastructure Cost**: Stay within $545/month budget
- **Cost per Workflow**: <$0.10 including all persistence overhead
- **Development Efficiency**: 50% reduction in debugging time
- **SLA Compliance**: 100% workflows complete under 3 seconds

## Conclusion

This production-grade state persistence design provides the foundation for reliable, scalable graph-based workflows while ensuring sub-3-second response times and enterprise-level security. The multi-tier architecture balances performance, cost, and reliability requirements for the Azure Universal RAG system.

**Next Steps**: Begin infrastructure deployment and state manager implementation to support the upcoming pydantic-graph migration.
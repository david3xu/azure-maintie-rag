// Data Services Module: Cosmos DB (Gremlin) Only - ML Workspace excluded temporarily
param environmentName string
param location string
param principalId string
param resourcePrefix string
param managedIdentityPrincipalId string

// Single configuration - CPU-ONLY OPTIMIZED (Azure for Students)
var config = {
  cosmosCapacityMode: 'Serverless'   // FREE: First 1M RU/s and 25GB storage/month
  cosmosRU: 0
}

// Azure Cosmos DB Account with Gremlin API for Knowledge Graphs
resource cosmosAccount 'Microsoft.DocumentDB/databaseAccounts@2023-04-15' = {
  name: 'cosmos-${resourcePrefix}-${environmentName}-${uniqueString(resourceGroup().id)}'
  location: location
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
    locations: [
      {
        locationName: location
        failoverPriority: 0
        isZoneRedundant: environmentName == 'production'
      }
    ]
    capabilities: concat([
      {
        name: 'EnableGremlin'
      }
    ], config.cosmosCapacityMode == 'Serverless' ? [
      {
        name: 'EnableServerless'
      }
    ] : [])
    enableAutomaticFailover: environmentName == 'production'
    enableMultipleWriteLocations: false
    isVirtualNetworkFilterEnabled: false
    virtualNetworkRules: []
    ipRules: []
    enableFreeTier: true  // ENABLE: Use free tier when available
    publicNetworkAccess: 'Enabled'
  }
  identity: {
    type: 'SystemAssigned'
  }
  tags: {
    Environment: environmentName
    Purpose: 'Knowledge graph storage using Gremlin API for maintenance relationships'
  }
}

// Gremlin Database for Knowledge Graphs
resource gremlinDatabase 'Microsoft.DocumentDB/databaseAccounts/gremlinDatabases@2023-04-15' = {
  parent: cosmosAccount
  name: 'maintie-rag-${environmentName}'
  properties: {
    resource: {
      id: 'maintie-rag-${environmentName}'
    }
  }
}

// Knowledge Graph Container for maintenance entities and relationships
resource knowledgeGraphContainer 'Microsoft.DocumentDB/databaseAccounts/gremlinDatabases/graphs@2023-04-15' = {
  parent: gremlinDatabase
  name: 'knowledge-graph-${environmentName}'
  properties: config.cosmosCapacityMode == 'Serverless' ? {
    resource: {
      id: 'knowledge-graph-${environmentName}'
      partitionKey: {
        paths: ['/partitionKey']
        kind: 'Hash'
      }
      indexingPolicy: {
        indexingMode: 'consistent'
        includedPaths: [
          {
            path: '/*'
          }
        ]
        excludedPaths: [
          {
            path: '/"_etag"/?'
          }
        ]
      }
    }
  } : {
    resource: {
      id: 'knowledge-graph-${environmentName}'
      partitionKey: {
        paths: ['/partitionKey']
        kind: 'Hash'
      }
      indexingPolicy: {
        indexingMode: 'consistent'
        includedPaths: [
          {
            path: '/*'
          }
        ]
        excludedPaths: [
          {
            path: '/"_etag"/?'
          }
        ]
      }
    }
    options: {
      throughput: config.cosmosRU
    }
  }
}

// RBAC assignments for managed identity
resource managedIdentityCosmosContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: cosmosAccount
  name: guid(cosmosAccount.id, managedIdentityPrincipalId, 'Cosmos DB Built-in Data Contributor')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '00482a5a-887f-4fb3-b363-3b7fe8e74483')
    principalId: managedIdentityPrincipalId
    principalType: 'ServicePrincipal'
  }
}

// RBAC assignments for current user (development)
resource cosmosContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(principalId)) {
  scope: cosmosAccount
  name: guid(cosmosAccount.id, principalId, 'Cosmos DB Built-in Data Contributor')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '00482a5a-887f-4fb3-b363-3b7fe8e74483')
    principalId: principalId
    principalType: 'User'
  }
}

// Diagnostics for monitoring
resource cosmosDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  scope: cosmosAccount
  name: 'cosmos-diagnostics'
  properties: {
    workspaceId: resourceId('Microsoft.OperationalInsights/workspaces', 'log-${resourcePrefix}-${environmentName}')
    logs: [
      {
        categoryGroup: 'allLogs'
        enabled: true
        retentionPolicy: {
          enabled: false
          days: 0
        }
      }
    ]
    metrics: [
      {
        category: 'AllMetrics'
        enabled: true
        retentionPolicy: {
          enabled: false
          days: 0
        }
      }
    ]
  }
}

// Azure ML Workspace for GNN Training and Inference
resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2023-04-01' = {
  name: 'ml-${resourcePrefix}-${environmentName}-${substring(uniqueString(resourceGroup().id), 0, 6)}'
  location: location
  properties: {
    friendlyName: 'Azure Universal RAG ML Workspace'
    description: 'Machine Learning workspace for GNN training and inference'
    storageAccount: resourceId('Microsoft.Storage/storageAccounts', 'st${take(replace(replace('${resourcePrefix}${environmentName}', '-', ''), '_', ''), 8)}${take(uniqueString(resourceGroup().id, resourcePrefix, environmentName), 10)}')
    keyVault: resourceId('Microsoft.KeyVault/vaults', 'kv-${take(replace('${resourcePrefix}${environmentName}', '-', ''), 12)}-${take(uniqueString(resourceGroup().id, resourcePrefix, environmentName), 8)}')
    applicationInsights: resourceId('Microsoft.Insights/components', 'appi-${resourcePrefix}-${environmentName}')
    publicNetworkAccess: 'Enabled'
    imageBuildCompute: 'ml-cluster-${environmentName}'
  }
  identity: {
    type: 'SystemAssigned'
  }
  tags: {
    Environment: environmentName
    Purpose: 'GNN training and inference for Universal RAG'
  }
}

// Note: ML Compute resources removed due to Azure for Students vCPU quota limitations
// Compute can be added later via Azure portal or CLI if quota is increased

// RBAC assignments for ML workspace
resource managedIdentityMLContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: mlWorkspace
  name: guid(mlWorkspace.id, managedIdentityPrincipalId, 'AzureML Data Scientist')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'f6c7c914-8db3-469d-8ca1-694a8f32e121')  // AzureML Data Scientist
    principalId: managedIdentityPrincipalId
    principalType: 'ServicePrincipal'
  }
}

// RBAC assignments for current user (development)
resource mlDataScientist 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(principalId)) {
  scope: mlWorkspace
  name: guid(mlWorkspace.id, principalId, 'AzureML Data Scientist')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'f6c7c914-8db3-469d-8ca1-694a8f32e121')  // AzureML Data Scientist
    principalId: principalId
    principalType: 'User'
  }
}

// Outputs
output cosmosAccountName string = cosmosAccount.name
output cosmosEndpoint string = cosmosAccount.properties.documentEndpoint
output cosmosResourceId string = cosmosAccount.id
output cosmosPrimaryKey string = cosmosAccount.listKeys().primaryMasterKey
output cosmosGremlinEndpoint string = 'wss://${cosmosAccount.name}.gremlin.cosmos.azure.com:443/'

output gremlinDatabaseName string = gremlinDatabase.name
output knowledgeGraphName string = knowledgeGraphContainer.name

// Azure ML workspace outputs
output mlWorkspaceName string = mlWorkspace.name
output mlWorkspaceId string = mlWorkspace.id
output mlWorkspaceEndpoint string = 'https://${mlWorkspace.name}.api.azureml.ms'
output mlComputeClusterName string = 'ml-cluster-${environmentName}'  // Placeholder - compute not deployed due to quota
output mlComputeInstanceName string = 'ml-instance-${environmentName}'  // Placeholder - compute not deployed due to quota
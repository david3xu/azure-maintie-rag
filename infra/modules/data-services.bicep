// Data Services Module: Cosmos DB (Gremlin) and Azure ML Workspace
param environmentName string
param location string
param principalId string
param resourcePrefix string
param managedIdentityPrincipalId string

// Environment-specific configuration
var environmentConfig = {
  development: {
    cosmosCapacityMode: 'Serverless'
    cosmosRU: 0
    mlComputeSize: 'Standard_DS3_v2'
    mlMinNodes: 0
    mlMaxNodes: 1
  }
  staging: {
    cosmosCapacityMode: 'Provisioned'
    cosmosRU: 1000
    mlComputeSize: 'Standard_DS3_v2'
    mlMinNodes: 0
    mlMaxNodes: 3
  }
  production: {
    cosmosCapacityMode: 'Provisioned'
    cosmosRU: 4000
    mlComputeSize: 'Standard_DS4_v2'
    mlMinNodes: 1
    mlMaxNodes: 10
  }
}

var config = environmentConfig[environmentName]

// Azure Cosmos DB Account with Gremlin API for Knowledge Graphs
resource cosmosAccount 'Microsoft.DocumentDB/databaseAccounts@2023-04-15' = {
  name: 'cosmos-${resourcePrefix}-${environmentName}-${uniqueString(resourceGroup().id)}'
  location: 'centralus'
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
    locations: [
      {
        locationName: 'centralus'
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
    enableFreeTier: false
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

// Azure ML Workspace for GNN Training  
resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2023-04-01' = {
  name: 'ml-${take(replace('${resourcePrefix}${environmentName}', '-', ''), 10)}-${take(uniqueString(resourceGroup().id, deployment().name), 8)}'
  location: 'centralus'
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    name: 'Basic'
    tier: 'Basic'
  }
  properties: {
    description: 'Azure ML Workspace for GNN training and model management'
    friendlyName: 'Universal RAG ML Workspace (${environmentName})'
    keyVault: resourceId('Microsoft.KeyVault/vaults', 'kv-maintieragst-bfyhcuxj')
    storageAccount: resourceId('Microsoft.Storage/storageAccounts', 'stmaintieroeeopj3ksg')
    applicationInsights: resourceId('Microsoft.Insights/components', 'appi-${resourcePrefix}-${environmentName}')
    publicNetworkAccess: 'Enabled'
    v1LegacyMode: false
  }
  tags: {
    Environment: environmentName
    Purpose: 'GNN model training for enhanced maintenance knowledge graph reasoning'
  }
}

// Compute Instance for ML development and experimentation
resource mlComputeInstance 'Microsoft.MachineLearningServices/workspaces/computes@2023-04-01' = if (environmentName != 'production' && !empty(principalId)) {
  parent: mlWorkspace
  name: 'compute-${environmentName}'
  location: 'centralus'
  properties: {
    computeType: 'ComputeInstance'
    properties: {
      vmSize: config.mlComputeSize
      applicationSharingPolicy: 'Personal'
      computeInstanceAuthorizationType: 'personal'
      enableNodePublicIp: true
      personalComputeInstanceSettings: {
        assignedUser: {
          objectId: principalId
          tenantId: subscription().tenantId
        }
      }
    }
  }
}

// Compute Cluster for scalable ML training
resource mlComputeCluster 'Microsoft.MachineLearningServices/workspaces/computes@2023-04-01' = {
  parent: mlWorkspace
  name: 'cluster-${environmentName}'
  location: 'centralus'
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: config.mlComputeSize
      vmPriority: 'Dedicated'
      scaleSettings: {
        minNodeCount: config.mlMinNodes
        maxNodeCount: config.mlMaxNodes
        nodeIdleTimeBeforeScaleDown: 'PT120S'
      }
      enableNodePublicIp: true
      isolatedNetwork: false
      osType: 'Linux'
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

resource managedIdentityMlContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: mlWorkspace
  name: guid(mlWorkspace.id, managedIdentityPrincipalId, 'AzureML Data Scientist')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'f6c7c914-8db3-469d-8ca1-694a8f32e121')
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

resource mlWorkspaceContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(principalId)) {
  scope: mlWorkspace
  name: guid(mlWorkspace.id, principalId, 'AzureML Data Scientist')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'f6c7c914-8db3-469d-8ca1-694a8f32e121')
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

// Outputs
output cosmosAccountName string = cosmosAccount.name
output cosmosEndpoint string = cosmosAccount.properties.documentEndpoint
output cosmosResourceId string = cosmosAccount.id
output cosmosPrimaryKey string = cosmosAccount.listKeys().primaryMasterKey
output cosmosGremlinEndpoint string = 'wss://${cosmosAccount.name}.gremlin.cosmos.azure.com:443/'

output gremlinDatabaseName string = gremlinDatabase.name
output knowledgeGraphName string = knowledgeGraphContainer.name

output mlWorkspaceName string = mlWorkspace.name
output mlWorkspaceId string = mlWorkspace.id
output mlWorkspaceEndpoint string = mlWorkspace.properties.workspaceId
output mlComputeClusterName string = mlComputeCluster.name
output mlComputeInstanceName string = environmentName != 'production' ? mlComputeInstance.name : ''
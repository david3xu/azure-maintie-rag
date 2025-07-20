// Azure Universal RAG Infrastructure - Core Version
// Core resources without Cosmos DB to avoid region availability issues

targetScope = 'resourceGroup'

// Parameters for environment-specific configuration
param environment string = 'dev'
param location string = resourceGroup().location
param resourcePrefix string = 'maintie'

// Deployment timestamp for unique resource naming
// Note: This parameter is available for future use but not currently used in resource names
// @description('Deployment timestamp for resource naming')
// param deploymentTimestamp string = utcNow('yyyyMMdd-HHmmss')

// Data-driven resource configuration by environment
var resourceConfig = {
  dev: {
    searchSku: 'basic'
    searchReplicas: 1
    searchPartitions: 1
    storageSku: 'Standard_LRS'
    storageAccessTier: 'Cool'
    keyVaultSku: 'standard'
    appInsightsSampling: 10
    cosmosThroughput: 400
    mlComputeInstances: 1
    openaiTokensPerMinute: 10000
    retentionDays: 30
  }
  staging: {
    searchSku: 'standard'
    searchReplicas: 1
    searchPartitions: 1
    storageSku: 'Standard_ZRS'
    storageAccessTier: 'Hot'
    keyVaultSku: 'standard'
    appInsightsSampling: 5
    cosmosThroughput: 800
    mlComputeInstances: 2
    openaiTokensPerMinute: 20000
    retentionDays: 60
  }
  prod: {
    searchSku: 'standard'
    searchReplicas: 2
    searchPartitions: 2
    storageSku: 'Standard_GRS'
    storageAccessTier: 'Hot'
    keyVaultSku: 'premium'
    appInsightsSampling: 1
    cosmosThroughput: 1600
    mlComputeInstances: 4
    openaiTokensPerMinute: 40000
    retentionDays: 90
  }
}

// Get current environment configuration
var currentConfig = resourceConfig[environment]

// Parameters for unique resource names
param storageAccountName string
param searchServiceName string
param keyVaultName string

// Azure Storage Account for Universal RAG data
resource storageAccount 'Microsoft.Storage/storageAccounts@2021-04-01' = {
  name: storageAccountName
  location: location
  sku: { name: currentConfig.storageSku }  // Data-driven
  kind: 'StorageV2'
  properties: {
    isHnsEnabled: true  // Hierarchical namespace for data lake
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    accessTier: currentConfig.storageAccessTier  // Data-driven
  }
}

// Blob service for Universal RAG data
resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2021-04-01' = {
  parent: storageAccount
  name: 'default'
}

// Storage container for Universal RAG data
resource blobContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2021-04-01' = {
  parent: blobService
  name: 'universal-rag-data'
  properties: {
    publicAccess: 'None'
  }
}

// Azure Cognitive Search for vector indices
resource searchService 'Microsoft.Search/searchServices@2020-08-01' = {
  name: searchServiceName
  location: location
  sku: { name: currentConfig.searchSku }  // Data-driven
  properties: {
    replicaCount: currentConfig.searchReplicas  // Data-driven
    partitionCount: currentConfig.searchPartitions  // Data-driven
    hostingMode: 'default'
  }
}

// Azure Key Vault for secrets management
resource keyVault 'Microsoft.KeyVault/vaults@2021-06-01-preview' = {
  name: keyVaultName
  location: location
  properties: {
    sku: {
      family: 'A'
      name: currentConfig.keyVaultSku  // Data-driven
    }
    tenantId: subscription().tenantId
    accessPolicies: []
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
  }
}

// Azure Cosmos DB for knowledge graph (Gremlin API) - Optional for dev
resource cosmosDB 'Microsoft.DocumentDB/databaseAccounts@2023-04-15' = if (environment == 'prod') {
  name: '${resourcePrefix}-${environment}-cosmos'
  location: location
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    capabilities: [
      { name: 'EnableGremlin' }
    ]
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
    locations: [
      {
        locationName: location
        failoverPriority: 0
        isZoneRedundant: true
      }
    ]
    enableFreeTier: false
  }
}

// Cosmos database for Universal RAG - Optional for dev
resource cosmosDatabase 'Microsoft.DocumentDB/databaseAccounts/gremlinDatabases@2023-04-15' = if (environment == 'prod') {
  parent: cosmosDB
  name: 'universal-rag-db'
  properties: {
    resource: {
      id: 'universal-rag-db'
    }
    options: {
      throughput: currentConfig.cosmosThroughput  // Data-driven
    }
  }
}

// Cosmos container for knowledge graph - Optional for dev
resource cosmosContainer 'Microsoft.DocumentDB/databaseAccounts/gremlinDatabases/graphs@2023-04-15' = if (environment == 'prod') {
  parent: cosmosDatabase
  name: 'knowledge-graph'
  properties: {
    resource: {
      id: 'knowledge-graph'
      partitionKey: {
        paths: ['/domain']
        kind: 'Hash'
      }
    }
  }
}

// Azure ML Workspace for GNN training - Simplified for dev
resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2023-04-01' = if (environment == 'prod') {
  name: '${resourcePrefix}-${environment}-ml'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: 'Universal RAG ML Workspace'
    description: 'Azure ML workspace for GNN training and model management'
    keyVault: keyVault.id
    storageAccount: storageAccount.id
    applicationInsights: applicationInsights.id
  }
}

// Azure Application Insights for monitoring
resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${resourcePrefix}-${environment}-appinsights'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalyticsWorkspace.id
    SamplingPercentage: currentConfig.appInsightsSampling  // Data-driven
  }
}

// Log Analytics Workspace for monitoring
resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${resourcePrefix}-${environment}-logs'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: currentConfig.retentionDays  // Data-driven
  }
}

// Outputs for deployment
output storageAccountName string = storageAccount.name
output searchServiceName string = searchService.name
output keyVaultName string = keyVault.name
output mlWorkspaceName string = (environment == 'prod') ? mlWorkspace.name : 'not-deployed'
output applicationInsightsName string = applicationInsights.name
output logAnalyticsWorkspaceName string = logAnalyticsWorkspace.name
output environmentConfig object = currentConfig  // Export configuration for reference
// Azure Universal RAG Infrastructure as Code
// Based on resume tech foundations and Universal RAG requirements

targetScope = 'resourceGroup'

// Parameters for environment-specific configuration
param environment string = 'dev'
param location string = resourceGroup().location
param resourcePrefix string = 'maintie'

// Azure Storage Account for Universal RAG data
resource storageAccount 'Microsoft.Storage/storageAccounts@2021-04-01' = {
  name: '${resourcePrefix}${environment}storage'
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    isHnsEnabled: true  // Hierarchical namespace for data lake
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
  }
}

// Storage container for Universal RAG data
resource blobContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2021-04-01' = {
  parent: storageAccount
  name: 'default/universal-rag-data'
  properties: {
    publicAccess: 'None'
  }
}

// Azure Cognitive Search for vector indices
resource searchService 'Microsoft.Search/searchServices@2020-08-01' = {
  name: '${resourcePrefix}-${environment}-search'
  location: location
  sku: { name: 'standard' }
  properties: {
    replicaCount: 1
    partitionCount: 1
    hostingMode: 'default'
  }
}

// Azure Cosmos DB for knowledge graphs
resource cosmosAccount 'Microsoft.DocumentDB/databaseAccounts@2021-04-15' = {
  name: '${resourcePrefix}-${environment}-cosmos'
  location: location
  properties: {
    capabilities: [{ name: 'EnableGremlin' }]  // Graph API
    databaseAccountOfferType: 'Standard'
    locations: [{
      locationName: location
      failoverPriority: 0
    }]
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
  }
}

// Cosmos DB database for Universal RAG
resource cosmosDatabase 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases@2021-04-15' = {
  parent: cosmosAccount
  name: 'universal-rag-db'
  properties: {
    resource: {
      id: 'universal-rag-db'
    }
  }
}

// Cosmos DB container for knowledge graphs
resource cosmosContainer 'Microsoft.DocumentDB/databaseAccounts/sqlDatabases/containers@2021-04-15' = {
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

// Azure Container Apps for deployment
resource containerApp 'Microsoft.App/containerApps@2022-03-01' = {
  name: '${resourcePrefix}-${environment}-app'
  location: location
  properties: {
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        allowInsecure: false
      }
    }
    template: {
      containers: [{
        name: 'universal-rag'
        image: 'universal-rag:latest'
        resources: {
          cpu: 1.0
          memory: '2Gi'
        }
        env: [{
          name: 'AZURE_STORAGE_ACCOUNT'
          value: storageAccount.name
        }, {
          name: 'AZURE_SEARCH_SERVICE'
          value: searchService.name
        }, {
          name: 'AZURE_COSMOS_ENDPOINT'
          value: cosmosAccount.properties.documentEndpoint
        }]
      }]
      scale: {
        minReplicas: 1
        maxReplicas: 10
      }
    }
  }
}

// Azure Key Vault for secrets management
resource keyVault 'Microsoft.KeyVault/vaults@2021-06-01-preview' = {
  name: '${resourcePrefix}-${environment}-kv'
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    accessPolicies: []
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
  }
}

// Outputs for deployment
output storageAccountName string = storageAccount.name
output searchServiceName string = searchService.name
output cosmosAccountName string = cosmosAccount.name
output containerAppName string = containerApp.name
output keyVaultName string = keyVault.name
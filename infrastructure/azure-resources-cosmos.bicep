// Azure Universal RAG Cosmos DB Infrastructure - Gremlin API for Knowledge Graphs
// Deploys: Cosmos DB Account, Gremlin Database, Knowledge Graph Container
// Based on existing codebase configuration patterns

targetScope = 'resourceGroup'

// Parameters for environment-specific configuration
param environment string = 'dev'
param location string = resourceGroup().location
param resourcePrefix string = 'maintie'

// Data-driven resource configuration by environment - from backend/config/settings.py
var resourceConfig = {
  dev: {
    cosmosThroughput: 400
    cosmosBackupPolicy: 'Periodic'
    cosmosConsistencyLevel: 'Session'
    enableFreeTier: true
    enableAnalyticalStorage: false
  }
  staging: {
    cosmosThroughput: 800
    cosmosBackupPolicy: 'Continuous'
    cosmosConsistencyLevel: 'Session'
    enableFreeTier: false
    enableAnalyticalStorage: true
  }
  prod: {
    cosmosThroughput: 1600
    cosmosBackupPolicy: 'Continuous'
    cosmosConsistencyLevel: 'BoundedStaleness'
    enableFreeTier: false
    enableAnalyticalStorage: true
  }
}

// Get current environment configuration
var currentConfig = resourceConfig[environment]

// Use same deterministic naming pattern as other templates
param deploymentToken string = uniqueString(resourceGroup().id, deployment().name)

// Deterministic resource naming - following existing pattern from codebase
var cosmosAccountName = '${resourcePrefix}-${environment}-cosmos-${take(deploymentToken, 6)}'

// Database and container names from backend/config/environments/*.env
var cosmosDatabaseName = 'universal-rag-db-${environment}'
var cosmosContainerName = 'knowledge-graph-${environment}'

// Cosmos DB Account with Gremlin API - based on backend/core/azure_cosmos/cosmos_gremlin_client.py
resource cosmosAccount 'Microsoft.DocumentDB/databaseAccounts@2021-04-15' = {
  name: cosmosAccountName
  location: location
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    capabilities: [
      { name: 'EnableGremlin' }  // Required for Gremlin API as used in cosmos_gremlin_client.py
    ]
    locations: [
      {
        locationName: location
        failoverPriority: 0
        isZoneRedundant: false
      }
    ]
    consistencyPolicy: {
      defaultConsistencyLevel: currentConfig.cosmosConsistencyLevel
      maxIntervalInSeconds: currentConfig.cosmosConsistencyLevel == 'BoundedStaleness' ? 86400 : null
      maxStalenessPrefix: currentConfig.cosmosConsistencyLevel == 'BoundedStaleness' ? 100000 : null
    }
    backupPolicy: {
      type: currentConfig.cosmosBackupPolicy
      periodicModeProperties: currentConfig.cosmosBackupPolicy == 'Periodic' ? {
        backupIntervalInMinutes: 240
        backupRetentionIntervalInHours: 8
      } : null
    }
    isVirtualNetworkFilterEnabled: false
    enableFreeTier: currentConfig.enableFreeTier
    enableAnalyticalStorage: currentConfig.enableAnalyticalStorage
  }
}

// Cosmos DB Gremlin Database - matches backend/config/settings.py cosmos_db_database_name
resource cosmosDatabase 'Microsoft.DocumentDB/databaseAccounts/gremlinDatabases@2021-04-15' = {
  parent: cosmosAccount
  name: cosmosDatabaseName
  properties: {
    resource: {
      id: cosmosDatabaseName
    }
    options: {
      throughput: currentConfig.cosmosThroughput
    }
  }
}

// Cosmos DB Gremlin Graph Container - matches backend/config/settings.py cosmos_db_container_name
resource cosmosGraph 'Microsoft.DocumentDB/databaseAccounts/gremlinDatabases/graphs@2021-04-15' = {
  parent: cosmosDatabase
  name: cosmosContainerName
  properties: {
    resource: {
      id: cosmosContainerName
      partitionKey: {
        paths: ['/partitionKey']  // Standard partition key for graph data
        kind: 'Hash'
      }
      defaultTtl: -1  // No automatic expiration
      uniqueKeyPolicy: {
        uniqueKeys: []
      }
    }
  }
}

// Outputs for verification and environment configuration
output cosmosAccountName string = cosmosAccount.name
output cosmosAccountId string = cosmosAccount.id
output cosmosEndpoint string = cosmosAccount.properties.documentEndpoint
output cosmosDatabaseName string = cosmosDatabase.name
output cosmosContainerName string = cosmosGraph.name
output deploymentToken string = deploymentToken
output environmentConfig object = currentConfig

// Connection string format for backend/config/environment_example.env
output cosmosConnectionStringFormat string = 'AccountEndpoint=${cosmosAccount.properties.documentEndpoint};AccountKey=<PRIMARY_KEY>;'
output cosmosGremlinEndpoint string = replace(replace(cosmosAccount.properties.documentEndpoint, 'https://', 'wss://'), ':443/', ':443/gremlin/')
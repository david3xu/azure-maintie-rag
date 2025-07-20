// Azure Universal RAG ML Infrastructure - Missing Services
// Deploys: ML Storage, Cosmos DB Gremlin, ML Workspace, Container Environment, Container App

targetScope = 'resourceGroup'

// Parameters for environment-specific configuration
param environment string = 'dev'
param location string = resourceGroup().location
param resourcePrefix string = 'maintie'

// Data-driven resource configuration by environment
var resourceConfig = {
  dev: {
    cosmosThroughput: 400
    cosmosBackupPolicy: 'Periodic'
    mlComputeInstances: 1
    mlVmSize: 'Standard_DS2_v2'
    containerCpuCores: '0.5'
    containerMemory: '1.0Gi'
    containerReplicas: 1
    storageSku: 'Standard_LRS'
    storageAccessTier: 'Cool'
  }
  staging: {
    cosmosThroughput: 800
    cosmosBackupPolicy: 'Continuous'
    mlComputeInstances: 2
    mlVmSize: 'Standard_DS3_v2'
    containerCpuCores: '1.0'
    containerMemory: '2.0Gi'
    containerReplicas: 2
    storageSku: 'Standard_ZRS'
    storageAccessTier: 'Hot'
  }
  prod: {
    cosmosThroughput: 1600
    cosmosBackupPolicy: 'Continuous'
    mlComputeInstances: 4
    mlVmSize: 'Standard_DS4_v2'
    containerCpuCores: '2.0'
    containerMemory: '4.0Gi'
    containerReplicas: 3
    storageSku: 'Standard_GRS'
    storageAccessTier: 'Hot'
  }
}

// Get current environment configuration
var currentConfig = resourceConfig[environment]

// Deterministic naming with environment and resource prefix
param deploymentToken string = uniqueString(resourceGroup().id, environment, resourcePrefix)

// Deterministic resource naming (shorter for storage account)
var mlStorageAccountName = '${resourcePrefix}${environment}ml${take(deploymentToken, 4)}'
var cosmosAccountName = '${resourcePrefix}-${environment}-cosmos-${take(deploymentToken, 6)}'
var mlWorkspaceName = '${resourcePrefix}-${environment}-ml-${take(deploymentToken, 6)}'
var containerEnvironmentName = '${resourcePrefix}-${environment}-env-${take(deploymentToken, 6)}'
var containerAppName = '${resourcePrefix}-${environment}-rag-app-${take(deploymentToken, 6)}'

// Parameters for existing resources
param existingStorageAccountName string
param existingKeyVaultName string
param existingAppInsightsName string = '${resourcePrefix}-${environment}-appinsights'
param existingLogAnalyticsName string = '${resourcePrefix}-${environment}-logs'

// Get existing resources from core deployment
resource existingStorageAccount 'Microsoft.Storage/storageAccounts@2021-04-01' existing = {
  name: existingStorageAccountName
}

resource existingKeyVault 'Microsoft.KeyVault/vaults@2021-06-01-preview' existing = {
  name: existingKeyVaultName
}

resource existingAppInsights 'Microsoft.Insights/components@2020-02-02' existing = {
  name: existingAppInsightsName
}

resource existingLogAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' existing = {
  name: existingLogAnalyticsName
}

// ML Storage Account (for ML workspace data)
resource mlStorageAccount 'Microsoft.Storage/storageAccounts@2021-04-01' = {
  name: mlStorageAccountName
  location: location
  sku: { name: currentConfig.storageSku }
  kind: 'StorageV2'
  properties: {
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    accessTier: currentConfig.storageAccessTier
  }
}

// Cosmos DB Account with Gremlin API
resource cosmosAccount 'Microsoft.DocumentDB/databaseAccounts@2021-04-15' = {
  name: cosmosAccountName
  location: location
  kind: 'GlobalDocumentDB'
  properties: {
    databaseAccountOfferType: 'Standard'
    capabilities: [
      { name: 'EnableGremlin' }
    ]
    locations: [
      {
        locationName: location
        failoverPriority: 0
        isZoneRedundant: false
      }
    ]
    backupPolicy: {
      type: currentConfig.cosmosBackupPolicy
      periodicModeProperties: currentConfig.cosmosBackupPolicy == 'Periodic' ? {
        backupIntervalInMinutes: 240
        backupRetentionIntervalInHours: 8
      } : null
    }
    isVirtualNetworkFilterEnabled: false
    enableFreeTier: environment == 'dev'
  }
}

// Cosmos DB Gremlin Database
resource cosmosDatabase 'Microsoft.DocumentDB/databaseAccounts/gremlinDatabases@2021-04-15' = {
  parent: cosmosAccount
  name: 'universal-rag-db-${environment}'
  properties: {
    resource: {
      id: 'universal-rag-db-${environment}'
    }
    options: {
      throughput: currentConfig.cosmosThroughput
    }
  }
}

// Cosmos DB Gremlin Graph
resource cosmosGraph 'Microsoft.DocumentDB/databaseAccounts/gremlinDatabases/graphs@2021-04-15' = {
  parent: cosmosDatabase
  name: 'knowledge-graph-${environment}'
  properties: {
    resource: {
      id: 'knowledge-graph-${environment}'
      partitionKey: {
        paths: ['/partitionKey']
        kind: 'Hash'
      }
    }
  }
}

// Azure ML Workspace
resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2021-04-01' = {
  name: mlWorkspaceName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    storageAccount: mlStorageAccount.id
    keyVault: existingKeyVault.id
    applicationInsights: existingAppInsights.id
    friendlyName: 'Universal RAG ML Workspace - ${toUpper(environment)}'
    description: 'ML workspace for Universal RAG GNN training and inference'
  }
}

// Container Apps Environment
resource containerEnvironment 'Microsoft.App/managedEnvironments@2022-03-01' = {
  name: containerEnvironmentName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: existingLogAnalytics.properties.customerId
        sharedKey: existingLogAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// Container App for RAG API
resource containerApp 'Microsoft.App/containerApps@2022-03-01' = {
  name: containerAppName
  location: location
  properties: {
    managedEnvironmentId: containerEnvironment.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
      }
      secrets: [
        {
          name: 'storage-connection-string'
          value: 'DefaultEndpointsProtocol=https;AccountName=${existingStorageAccount.name};AccountKey=${existingStorageAccount.listKeys().keys[0].value};EndpointSuffix=core.windows.net'
        }
        {
          name: 'cosmos-connection-string'
          value: cosmosAccount.listConnectionStrings().connectionStrings[0].connectionString
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'universal-rag-api'
          image: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest' // Placeholder - replace with your image
          resources: {
            cpu: json(currentConfig.containerCpuCores)
            memory: currentConfig.containerMemory
          }
          env: [
            {
              name: 'AZURE_ENVIRONMENT'
              value: environment
            }
            {
              name: 'AZURE_STORAGE_CONNECTION_STRING'
              secretRef: 'storage-connection-string'
            }
            {
              name: 'AZURE_COSMOS_DB_CONNECTION_STRING'
              secretRef: 'cosmos-connection-string'
            }
            {
              name: 'AZURE_ML_WORKSPACE_NAME'
              value: mlWorkspaceName
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: currentConfig.containerReplicas
      }
    }
  }
}

// Outputs for verification
output mlStorageAccountName string = mlStorageAccount.name
output cosmosAccountName string = cosmosAccount.name
output mlWorkspaceName string = mlWorkspace.name
output containerEnvironmentName string = containerEnvironment.name
output containerAppName string = containerApp.name
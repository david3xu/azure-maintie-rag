@description('Environment (dev, staging, prod)')
param environment string = 'dev'

@description('Location for all resources')
param location string = resourceGroup().location

// Resource group name is available via resourceGroup().name

// Variables
var resourcePrefix = 'maintie-${environment}'
var mlWorkspaceName = '${resourcePrefix}-ml'
var appInsightsName = '${resourcePrefix}-app-insights'
var containerAppName = '${resourcePrefix}-rag-app'
var containerEnvName = '${resourcePrefix}-env'
// ML storage account will be created separately for compatibility

// ML Storage Account (separate from main storage, no HNS)
resource mlStorageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: '${resourcePrefix}mlstorage'
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    allowCrossTenantReplication: false
    isHnsEnabled: false  // Required for Azure ML compatibility
  }
}

// ML Workspace
resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: mlWorkspaceName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: 'Universal RAG ML Workspace'
    description: 'Azure ML workspace for Universal RAG GNN training and model management'
    storageAccount: mlStorageAccount.id
    keyVault: resourceId('Microsoft.KeyVault/vaults', '${resourcePrefix}-kv')
    applicationInsights: resourceId('Microsoft.Insights/components', appInsightsName)
    containerRegistry: null
    discoveryUrl: 'https://${location}.api.azureml.ms/'
    hbiWorkspace: false
    allowPublicAccessWhenBehindVnet: false
    publicNetworkAccess: 'Enabled'
    v1LegacyMode: false
  }
}

// Application Insights
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: resourceId('Microsoft.OperationalInsights/workspaces', '${resourcePrefix}-laworkspace')
    IngestionMode: 'LogAnalytics'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

// Log Analytics Workspace (required for Application Insights)
resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${resourcePrefix}-laworkspace'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
  }
}

// Container Apps Environment
resource containerEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: containerEnvName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalyticsWorkspace.properties.customerId
        sharedKey: logAnalyticsWorkspace.listKeys().primarySharedKey
      }
    }
  }
}

// Container App
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: containerAppName
  location: location
  properties: {
    managedEnvironmentId: containerEnvironment.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        allowInsecure: false
      }
    }
          template: {
        containers: [
          {
            name: 'rag-app'
            image: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
            env: [
              {
                name: 'AZURE_ENVIRONMENT'
                value: environment
              }
              {
                name: 'AZURE_USE_MANAGED_IDENTITY'
                value: 'true'
              }
            ]
          }
        ]
        scale: {
          minReplicas: 1
          maxReplicas: 3
        }
      }
  }
}

// Outputs
output mlWorkspaceName string = mlWorkspace.name
output appInsightsName string = appInsights.name
output containerAppName string = containerApp.name
output containerAppUrl string = containerApp.properties.configuration.ingress.fqdn
output logAnalyticsWorkspaceName string = logAnalyticsWorkspace.name
output mlStorageAccountName string = mlStorageAccount.name
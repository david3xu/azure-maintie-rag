// Hosting Services Module: Container Apps, Container Registry (FIXED)
param environmentName string
param location string
param resourcePrefix string

// Service endpoints from other modules
param openaiEndpoint string
param searchEndpoint string
param cosmosEndpoint string
@secure()
param cosmosKey string
param storageAccountName string
param keyVaultName string
@secure()
param appInsightsConnectionString string

// Container image names - provided by azd container building
@description('Backend container image name with tag (e.g., azure-maintie-rag/backend-prod:azd-deploy-123456)')
param backendImageName string = ''

@description('Frontend container image name with tag (e.g., azure-maintie-rag/frontend-prod:azd-deploy-123456)')
param frontendImageName string = ''

// Single configuration - FREE TIER OPTIMIZED (Cost Savings)
var config = {
  containerCpu: '0.25' // MINIMUM: Reduce CPU allocation
  containerMemory: '0.5Gi' // MINIMUM: Reduce memory allocation
  minReplicas: 0 // FREE: Scale to zero when not used
  maxReplicas: 1 // MINIMAL: Single replica to save costs
  registrySku: 'Basic' // CHEAPEST: Basic tier
}

// Get references to existing resources
resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: 'id-${resourcePrefix}-${environmentName}'
}

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' existing = {
  name: 'log-${resourcePrefix}-${environmentName}'
}

// Container Registry for hosting container images
// Supported in all Azure subscription types including Azure for Students
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: 'cr${replace(resourcePrefix, '-', '')}${environmentName}${substring(uniqueString(resourceGroup().id, resourcePrefix, environmentName), 0, 6)}'
  location: location
  sku: {
    name: config.registrySku
  }
  properties: {
    adminUserEnabled: true
    anonymousPullEnabled: false
    dataEndpointEnabled: false
    encryption: {
      status: 'disabled'
    }
    networkRuleBypassOptions: 'AzureServices'
    policies: {
      exportPolicy: {
        status: 'enabled'
      }
      retentionPolicy: {
        status: 'disabled'
        days: 7
      }
      trustPolicy: {
        status: 'disabled'
        type: 'Notary'
      }
    }
    publicNetworkAccess: 'Enabled'
    zoneRedundancy: 'Disabled'
  }
  tags: {
    Environment: environmentName
    Purpose: 'Container registry for Universal RAG images'
  }
}

// Container Apps Environment
resource containerEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: 'cae-${resourcePrefix}-${environmentName}'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
    zoneRedundant: environmentName == 'production'
  }
  tags: {
    Environment: environmentName
    Purpose: 'Container Apps hosting environment for Universal RAG'
  }
}

// Backend Container App
resource backendApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'ca-backend-${resourcePrefix}-${environmentName}'
  location: location
  tags: {
    Environment: environmentName
    Purpose: 'Backend API service for Universal RAG'
    'azd-service-name': 'backend'
  }
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: containerEnvironment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
        allowInsecure: true // Allow for initial setup
        traffic: [
          {
            weight: 100
            latestRevision: true
          }
        ]
      }
      registries: [
        {
          server: containerRegistry.properties.loginServer
          username: containerRegistry.name
          passwordSecretRef: 'container-registry-password'
        }
      ]
      secrets: [
        {
          name: 'app-insights-connection-string'
          value: appInsightsConnectionString
        }
        {
          name: 'container-registry-password'
          value: containerRegistry.listCredentials().passwords[0].value
        }
        {
          name: 'azure-cosmos-key'
          value: cosmosKey
        }
      ]
    }
    template: {
      containers: [
        {
          image: !empty(backendImageName)
            ? backendImageName
            : 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
          name: 'backend'
          env: [
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: openaiEndpoint
            }
            {
              name: 'AZURE_SEARCH_ENDPOINT'
              value: searchEndpoint
            }
            {
              name: 'AZURE_COSMOS_ENDPOINT'
              value: cosmosEndpoint
            }
            {
              name: 'AZURE_STORAGE_ACCOUNT'
              value: storageAccountName
            }
            {
              name: 'AZURE_KEY_VAULT_NAME'
              value: keyVaultName
            }
            {
              name: 'AZURE_CLIENT_ID'
              value: managedIdentity.properties.clientId
            }
            {
              name: 'AZURE_TENANT_ID'
              value: subscription().tenantId
            }
            {
              name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
              secretRef: 'app-insights-connection-string'
            }
            {
              name: 'PORT'
              value: '8000'
            }
            {
              name: 'USE_MANAGED_IDENTITY'
              value: 'false'
            }
            {
              name: 'AZURE_SEARCH_INDEX'
              value: 'maintie-prod-index'
            }
            {
              name: 'AZURE_COSMOS_DATABASE_NAME'
              value: 'maintie-rag-prod'
            }
            {
              name: 'AZURE_COSMOS_GRAPH_NAME'
              value: 'knowledge-graph-prod'
            }
            {
              name: 'AZURE_STORAGE_CONTAINER'
              value: 'maintie-prod-data'
            }
            {
              name: 'AZURE_OPENAI_DEPLOYMENT_NAME'
              value: 'gpt-4.1-mini'
            }
            {
              name: 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME'
              value: 'text-embedding-ada-002'
            }
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: openaiEndpoint
            }
            {
              name: 'AZURE_SEARCH_ENDPOINT'
              value: searchEndpoint
            }
            {
              name: 'AZURE_COSMOS_ENDPOINT'
              value: cosmosEndpoint
            }
            {
              name: 'AZURE_COSMOS_KEY'
              secretRef: 'azure-cosmos-key'
            }
            
          ]
          resources: {
            cpu: json(config.containerCpu)
            memory: config.containerMemory
          }
        }
      ]
      scale: {
        minReplicas: config.minReplicas
        maxReplicas: config.maxReplicas
      }
    }
  }
}

// Frontend Container App
resource frontendApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'ca-frontend-${resourcePrefix}-${environmentName}'
  location: location
  tags: {
    Environment: environmentName
    Purpose: 'Frontend UI service for Universal RAG'
    'azd-service-name': 'frontend'
  }
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: containerEnvironment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 3000
        transport: 'http'
        allowInsecure: true
        traffic: [
          {
            weight: 100
            latestRevision: true
          }
        ]
      }
      registries: [
        {
          server: containerRegistry.properties.loginServer
          username: containerRegistry.name
          passwordSecretRef: 'container-registry-password'
        }
      ]
      secrets: [
        {
          name: 'container-registry-password'
          value: containerRegistry.listCredentials().passwords[0].value
        }
      ]
    }
    template: {
      containers: [
        {
          image: !empty(frontendImageName)
            ? frontendImageName
            : 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
          name: 'frontend'
          env: [
            {
              name: 'VITE_API_URL'
              value: 'https://${backendApp.properties.configuration.ingress.fqdn}'
            }
            {
              name: 'PORT'
              value: '3000'
            }
          ]
          resources: {
            cpu: json(config.containerCpu)
            memory: config.containerMemory
          }
        }
      ]
      scale: {
        minReplicas: config.minReplicas
        maxReplicas: config.maxReplicas
      }
    }
  }
}

// Outputs
output containerRegistryName string = containerRegistry.name
output containerRegistryLoginServer string = containerRegistry.properties.loginServer
output backendAppUrl string = 'https://${backendApp.properties.configuration.ingress.fqdn}'
output frontendAppUrl string = 'https://${frontendApp.properties.configuration.ingress.fqdn}'
output containerEnvironmentId string = containerEnvironment.id

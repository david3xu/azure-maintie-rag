// Hosting Services Module: Container Apps, Container Registry
param environmentName string
param location string
param principalId string
param resourcePrefix string

// Service endpoints from other modules
param openaiEndpoint string
param searchEndpoint string
param cosmosEndpoint string
param storageAccountName string
param keyVaultName string
param appInsightsConnectionString string

// Container image names (passed from azd)
param backendImageName string = '${SERVICE_BACKEND_IMAGE_NAME}'
param frontendImageName string = '${SERVICE_FRONTEND_IMAGE_NAME}'

// Environment-specific configuration
var environmentConfig = {
  development: {
    containerCpu: '0.5'
    containerMemory: '1Gi'
    minReplicas: 1
    maxReplicas: 3
    registrySku: 'Basic'
  }
  staging: {
    containerCpu: '1.0'
    containerMemory: '2Gi'
    minReplicas: 1
    maxReplicas: 5
    registrySku: 'Standard'
  }
  prod: {
    containerCpu: '2.0'
    containerMemory: '4Gi'
    minReplicas: 2
    maxReplicas: 10
    registrySku: 'Premium'
  }
}

var config = environmentConfig[environmentName]

// Get references to existing resources
resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' existing = {
  name: 'id-${resourcePrefix}-${environmentName}'
}

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' existing = {
  name: 'log-${resourcePrefix}-${environmentName}'
}

// Azure Container Registry
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: 'cr${take(replace(replace('${resourcePrefix}${environmentName}', '-', ''), '_', ''), 10)}${take(uniqueString(resourceGroup().id), 10)}'
  location: location
  sku: {
    name: config.registrySku
  }
  properties: {
    adminUserEnabled: false
    publicNetworkAccess: 'Enabled'
    zoneRedundancy: environmentName == 'production' ? 'Enabled' : 'Disabled'
  }
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  tags: {
    Environment: environmentName
    Purpose: 'Container image storage for Universal RAG backend'
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
        allowInsecure: environmentName == 'development'
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
          identity: managedIdentity.id
        }
      ]
      secrets: [
        {
          name: 'app-insights-connection-string'
          value: appInsightsConnectionString
        }
      ]
    }
    template: {
      containers: [
        {
          image: '${containerRegistry.properties.loginServer}/${backendImageName}'
          name: 'backend'
          env: [
            // Azure Service Endpoints
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
            // Application Configuration
            {
              name: 'ENVIRONMENT'
              value: environmentName
            }
            {
              name: 'AZURE_CLIENT_ID'
              value: managedIdentity.properties.clientId
            }
            {
              name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
              secretRef: 'app-insights-connection-string'
            }
            // Model Deployments
            {
              name: 'OPENAI_MODEL_DEPLOYMENT'
              value: 'gpt-4o'
            }
            {
              name: 'EMBEDDING_MODEL_DEPLOYMENT'
              value: 'text-embedding-ada-002'
            }
            // Database Configuration
            {
              name: 'COSMOS_DATABASE_NAME'
              value: 'maintie-rag-${environmentName}'
            }
            {
              name: 'COSMOS_GRAPH_NAME'
              value: 'knowledge-graph-${environmentName}'
            }
            {
              name: 'SEARCH_INDEX_NAME'
              value: 'maintie-${environmentName}-index'
            }
            // Container Configuration
            {
              name: 'PORT'
              value: '8000'
            }
            {
              name: 'WORKERS'
              value: environmentName == 'production' ? '4' : '2'
            }
          ]
          resources: {
            cpu: json(config.containerCpu)
            memory: config.containerMemory
          }
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/health'
                port: 8000
                scheme: 'HTTP'
              }
              initialDelaySeconds: 30
              periodSeconds: 30
              timeoutSeconds: 10
              failureThreshold: 3
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/health'
                port: 8000
                scheme: 'HTTP'
              }
              initialDelaySeconds: 10
              periodSeconds: 10
              timeoutSeconds: 5
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: config.minReplicas
        maxReplicas: config.maxReplicas
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: environmentName == 'production' ? '100' : '50'
              }
            }
          }
          {
            name: 'cpu-scaling'
            custom: {
              type: 'cpu'
              metadata: {
                type: 'Utilization'
                value: '70'
              }
            }
          }
        ]
      }
    }
  }
  tags: {
    Environment: environmentName
    Purpose: 'Universal RAG FastAPI backend application'
    'azd-service-name': 'backend'
  }
}

// Frontend Container App
resource frontendApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'ca-frontend-${resourcePrefix}-${environmentName}'
  location: location
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
        allowInsecure: environmentName == 'development'
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
          identity: managedIdentity.id
        }
      ]
    }
    template: {
      containers: [
        {
          image: '${containerRegistry.properties.loginServer}/${frontendImageName}'
          name: 'frontend'
          env: [
            {
              name: 'VITE_API_URL'
              value: 'https://${backendApp.properties.configuration.ingress.fqdn}'
            }
            {
              name: 'NODE_ENV'
              value: environmentName == 'development' ? 'development' : 'production'
            }
          ]
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 5
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '30'
              }
            }
          }
        ]
      }
    }
  }
  tags: {
    Environment: environmentName
    Purpose: 'Universal RAG React frontend application'
    'azd-service-name': 'frontend'
  }
}

// RBAC for Container Registry
resource registryPullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: containerRegistry
  name: guid(containerRegistry.id, managedIdentity.id, 'AcrPull')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource registryPushRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(principalId)) {
  scope: containerRegistry
  name: guid(containerRegistry.id, principalId, 'AcrPush')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '8311e382-0749-4cb8-b61a-304f252e45ec')
    principalId: principalId
    principalType: 'User'
  }
}

// Container Apps Environment Diagnostics
resource environmentDiagnostics 'Microsoft.Insights/diagnosticSettings@2021-05-01-preview' = {
  scope: containerEnvironment
  name: 'environment-diagnostics'
  properties: {
    workspaceId: logAnalytics.id
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
  }
}

// Outputs
output registryName string = containerRegistry.name
output registryLoginServer string = containerRegistry.properties.loginServer
output registryId string = containerRegistry.id

output containerEnvironmentName string = containerEnvironment.name
output containerEnvironmentId string = containerEnvironment.id
output containerEnvironmentDefaultDomain string = containerEnvironment.properties.defaultDomain

output backendAppName string = backendApp.name
output backendUri string = 'https://${backendApp.properties.configuration.ingress.fqdn}'
output backendFqdn string = backendApp.properties.configuration.ingress.fqdn

output frontendAppName string = frontendApp.name
output frontendUri string = 'https://${frontendApp.properties.configuration.ingress.fqdn}'
output frontendFqdn string = frontendApp.properties.configuration.ingress.fqdn
// Alternative Cognitive Services that work without approval
param environmentName string
param location string
param principalId string
param resourcePrefix string
param managedIdentityPrincipalId string

// CognitiveServices multi-service (works without approval)
resource cognitiveServices 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: 'cs-${resourcePrefix}-${environmentName}-${uniqueString(resourceGroup().id)}'
  location: location
  kind: 'CognitiveServices'  // Multi-service without approval needed
  sku: {
    name: 'S0'  // Standard tier
  }
  properties: {
    customSubDomainName: 'cs-${resourcePrefix}-${environmentName}-${uniqueString(resourceGroup().id)}'
    publicNetworkAccess: 'Enabled'
  }
  tags: {
    Environment: environmentName
    Purpose: 'Multi-service Cognitive Services - Azure for Students compatible'
  }
}

// Text Analytics service (works without approval)
resource textAnalytics 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: 'ta-${resourcePrefix}-${environmentName}-${uniqueString(resourceGroup().id)}'
  location: location
  kind: 'TextAnalytics'
  sku: {
    name: 'S'  // Standard tier
  }
  properties: {
    customSubDomainName: 'ta-${resourcePrefix}-${environmentName}-${uniqueString(resourceGroup().id)}'
    publicNetworkAccess: 'Enabled'
  }
  tags: {
    Environment: environmentName
    Purpose: 'Text Analytics - Azure for Students compatible'
  }
}

// RBAC for managed identity access
resource managedIdentityCognitiveUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: cognitiveServices
  name: guid(cognitiveServices.id, managedIdentityPrincipalId, 'Cognitive Services User')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'a97b65f3-24c7-4388-baec-2e87135dc908')
    principalId: managedIdentityPrincipalId
    principalType: 'ServicePrincipal'
  }
}

// RBAC for user access (development)
resource cognitiveServicesUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(principalId)) {
  scope: cognitiveServices
  name: guid(cognitiveServices.id, principalId, 'Cognitive Services User')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'a97b65f3-24c7-4388-baec-2e87135dc908')
    principalId: principalId
    principalType: 'User'
  }
}

// Outputs
output cognitiveServicesEndpoint string = cognitiveServices.properties.endpoint
output cognitiveServicesName string = cognitiveServices.name
output textAnalyticsEndpoint string = textAnalytics.properties.endpoint
output textAnalyticsName string = textAnalytics.name
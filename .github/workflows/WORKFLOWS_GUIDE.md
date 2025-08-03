# GitHub Actions Workflows

This directory contains the automated workflows for the Azure Universal RAG system.

## Current Workflows

### `ci.yml` - Continuous Integration
- **Triggers**: Push to any branch, Pull Requests
- **Purpose**: Run unit tests, linting, code validation
- **Status**: ✅ Active (24/24 tests passing)

### `docker.yml` - Container Security
- **Triggers**: Push to main/develop branches
- **Purpose**: Build containers, security scanning
- **Status**: ✅ Active

### `azure-dev.yml` - Azure Deployment
- **Triggers**: Push to main/develop branches
- **Purpose**: Deploy to Azure using `azd up`
- **Status**: 🔧 **Created by `azd pipeline config`**
- **Environments**:
  - `develop` → staging
  - `main` → production

## Setup

The Azure deployment workflow is automatically configured. To set it up:

```bash
# Run this once to configure CI/CD
azd pipeline config
```

This command automatically:
- Creates the `azure-dev.yml` workflow
- Sets up Azure service principal
- Configures GitHub secrets
- Tests the deployment

## No Manual Configuration Needed!

Unlike traditional CI/CD setups, this system requires **zero manual secret configuration**. Everything is handled automatically by Azure Developer CLI.

## Monitoring

- **GitHub Actions**: View workflow runs in the Actions tab
- **Azure Portal**: Monitor deployed resources
- **Logs**: Use `azd logs` for deployment logs

## Architecture

```
Code Push → CI Tests → Azure Deployment
     ↓         ↓            ↓
  (ci.yml) (docker.yml) (azure-dev.yml)
```

The workflows are designed to be:
- **Simple**: Minimal configuration
- **Secure**: Automated secret management
- **Reliable**: Infrastructure as Code with Bicep
- **Fast**: Efficient deployment with azd

## Troubleshooting

If deployments fail:
1. Check the Actions tab for detailed logs
2. Run `azd logs` locally
3. Verify with `azd env get-values`

For setup issues:
1. Ensure `azd auth login` is successful
2. Re-run `azd pipeline config` if needed
3. Check Azure permissions

## Files Not to Modify

These files are managed automatically by `azd`:
- `azure-dev.yml` (created by azd pipeline config)
- Any files in `.azure/` directory

Modify these files manually:
- `ci.yml` (for test configuration)
- `docker.yml` (for container builds)

The automated approach ensures consistency and reduces configuration errors! 🚀

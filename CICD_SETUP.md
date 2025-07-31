# 🚀 CI/CD Setup with Azure Developer CLI (azd)

**Simple, Automatic, Zero Manual Configuration!**

## Quick Setup (3 commands!)

```bash
# 1. Login to Azure (if not already logged in)
azd auth login

# 2. Let azd configure CI/CD automatically
azd pipeline config

# 3. Done! 🎉
```

That's it! Your CI/CD pipeline is now fully configured.

## What Just Happened?

When you run `azd pipeline config`, it automatically:

✅ **Creates a service principal** in Azure AD  
✅ **Sets up GitHub secrets** in your repository  
✅ **Creates GitHub Actions workflow** (`.github/workflows/azure-dev.yml`)  
✅ **Configures staging and production environments**  
✅ **Tests the deployment** to make sure everything works  

**Zero manual secret configuration needed!**

## How It Works

### Your Workflow Now:
1. **Push code** to `main` or `develop` branch
2. **GitHub Actions** automatically triggered
3. **Unit tests run** (using existing `ci.yml`)
4. **Azure deployment** happens automatically with `azd up`
5. **Services deployed** to appropriate environment:
   - `develop` branch → **staging** environment
   - `main` branch → **production** environment

### What Gets Deployed:
- **Azure infrastructure** (from `/infra/` Bicep templates)
- **Backend service** (containerized from `/backend/`)
- **All Azure services** (OpenAI, Search, Cosmos DB, Storage, etc.)

## Environments

| Branch | Environment | Resource Group |
|--------|-------------|----------------|
| `develop` | staging | `rg-maintie-rag-staging` |
| `main` | production | `rg-maintie-rag-production` |

Each environment is completely isolated with its own Azure resources.

## Monitoring Your Deployments

### GitHub Actions
- Go to your repo → **Actions** tab
- See deployment status and logs
- Each push triggers automatic deployment

### Azure Portal
- **Staging**: Resource group `rg-maintie-rag-staging`
- **Production**: Resource group `rg-maintie-rag-production`
- View all deployed services, logs, and metrics

### Local Commands
```bash
# Check deployment status
azd env list

# View deployed service URLs
azd env get-values

# View logs
azd logs

# Deploy manually (if needed)
azd up
```

## Architecture

```
GitHub Push → GitHub Actions → azd up → Azure Deployment
     ↓              ↓             ↓            ↓
   Code          Unit Tests    Infrastructure  Services
  Changes        (ci.yml)      (Bicep)        (Container)
```

## Benefits of This Approach

### ✅ **Zero Configuration**
- No manual secrets to manage
- No complex authentication setup
- No environment variables to configure

### ✅ **Secure by Default**
- Service principal with minimal required permissions
- Secrets managed automatically by Azure
- Environment isolation

### ✅ **Consistent Environments**
- Same process locally (`azd up`) and in CI/CD
- Infrastructure as Code (Bicep templates)
- Reproducible deployments

### ✅ **Developer Friendly**
- Same commands work everywhere
- Easy debugging with `azd logs`
- Simple rollback with `azd down` + `azd up`

## Troubleshooting

### If `azd pipeline config` fails:
```bash
# Make sure you're logged in
azd auth login --check-status

# Make sure you're in the right directory
ls azure.yaml  # Should exist

# Check Azure permissions
az account show
```

### If deployment fails:
```bash
# Check logs
azd logs

# Check environment status
azd env get-values

# Re-deploy
azd up
```

### If you need to reconfigure:
```bash
# Clean and restart
azd pipeline config --force
```

## What's Next?

After running `azd pipeline config`:

1. **Push to develop** → triggers staging deployment
2. **Push to main** → triggers production deployment  
3. **Monitor in GitHub Actions** and Azure Portal
4. **Scale and iterate** as needed

Your Azure RAG system is now fully automated! 🎯

## Files Created by azd

After running `azd pipeline config`, you'll see:
- `.github/workflows/azure-dev.yml` - The automated workflow
- Additional environment files in `.azure/`

**Don't modify these manually** - they're managed by azd.

## Local Development

You can still deploy locally anytime:
```bash
# Deploy to your current environment
azd up

# Switch environments
azd env select staging
azd up
```

The beauty of `azd` is consistency between local and CI/CD! 🚀
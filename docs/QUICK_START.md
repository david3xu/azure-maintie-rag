# Quick Start Guide

## 🚀 Getting Started with Azure Universal RAG

This guide provides quick access to all documentation and helps you get up and running with the Azure Universal RAG system.

## 📋 **Quick Setup Checklist**

### **1. Prerequisites**
- [ ] Azure subscription with required services
- [ ] Python 3.10+ environment
- [ ] Git repository cloned

### **2. Initial Setup**
```bash
# 1. Configure Azure services
# Follow: AZURE_SETUP_GUIDE.md

# 2. Install dependencies
cd backend && pip install -r requirements.txt

# 3. Start the system
make dev
```

### **3. Data Preparation**
```bash
# Process your raw data
cd backend && make data-prep

# Verify everything is working
# Follow: docs/testing/system_verification_guide.md
```

### **4. Test the System**
```bash
# Health check
curl -s http://localhost:8000/api/v1/health

# Test a query
curl -s -X POST "http://localhost:8000/api/v1/query/universal" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are common maintenance issues?", "domain": "general"}' | jq .
```

## 📚 **Documentation Navigation**

### **🔧 Setup & Configuration**
- **[AZURE_SETUP_GUIDE.md](../AZURE_SETUP_GUIDE.md)** - Quick Azure setup
- **[AZURE_UNIVERSAL_RAG_DOCUMENTATION.md](../AZURE_UNIVERSAL_RAG_DOCUMENTATION.md)** - Complete implementation guide

### **🔄 Workflows**
- **[Data Preparation Workflow](workflows/data_preparation_workflow.md)** - How `make data-prep` works
- **[Data Preparation Architecture](workflows/data_preparation_architecture.md)** - Technical architecture details

### **🧪 Testing & Verification**
- **[System Verification Guide](testing/system_verification_guide.md)** - Complete testing procedures

## 🎯 **Common Tasks**

### **Add New Data**
1. Place markdown files in `backend/data/raw/`
2. Run `make data-prep`
3. Verify with testing guide

### **Test Queries**
```bash
# Universal query
curl -s -X POST "http://localhost:8000/api/v1/query/universal" \
  -H "Content-Type: application/json" \
  -d '{"query": "your question here", "domain": "general"}' | jq .

# Streaming query
curl -s -X POST "http://localhost:8000/api/v1/query/streaming" \
  -H "Content-Type: application/json" \
  -d '{"query": "your question here", "domain": "general"}' | jq .
```

### **Check System Status**
```bash
# Health check
curl -s http://localhost:8000/api/v1/health

# System info
curl -s http://localhost:8000/api/v1/info | jq .

# Domain status
curl -s "http://localhost:8000/api/v1/domain/general/status" | jq .
```

## 🐛 **Troubleshooting**

### **Common Issues**

| Issue | Solution | Documentation |
|-------|----------|---------------|
| FastAPI server not starting | Check dependencies and environment | [Setup Guide](../AZURE_SETUP_GUIDE.md) |
| Azure services not connecting | Verify `.env` configuration | [Setup Guide](../AZURE_SETUP_GUIDE.md) |
| Query endpoints failing | Check server logs and restart | [Verification Guide](testing/system_verification_guide.md) |
| Data preparation errors | Verify raw data format | [Workflow Guide](workflows/data_preparation_workflow.md) |

### **Quick Debug Commands**
```bash
# Check server status
curl -s http://localhost:8000/api/v1/health

# Check Azure services
curl -s http://localhost:8000/api/v1/info | jq '.azure_status'

# Check raw data
ls -la backend/data/raw/

# Check logs
tail -f backend.log
```

## 📊 **Expected Results**

### **After `make data-prep`:**
- ✅ Processing time: ~11-12 seconds
- ✅ Documents processed: Files from `data/raw/`
- ✅ Search index created: `rag-index-general`
- ✅ Blob storage: Documents uploaded
- ✅ Metadata: Stored in Cosmos DB

### **After Query Tests:**
- ✅ Query responses: Based on your data
- ✅ Search results: Relevant document snippets
- ✅ Processing time: <5 seconds per query
- ✅ Error handling: Graceful degradation

## 🚀 **Next Steps**

1. **Add More Data**: Place additional markdown files in `data/raw/`
2. **Test Different Queries**: Try various types of questions
3. **Scale Up**: Process larger datasets
4. **Deploy to Production**: Follow deployment guides
5. **Monitor Performance**: Set up monitoring and alerting

## 📞 **Support**

- **Documentation**: All guides are linked above
- **Issues**: Check troubleshooting sections in each guide
- **Architecture**: See technical architecture documentation
- **Testing**: Follow verification guide for comprehensive testing

---

**Last Updated**: July 2025
**Version**: 2.0.0
**Status**: Production Ready ✅
#!/usr/bin/env python3
"""
Environment Validation Script
Quick test for Azure Universal RAG system setup
"""

import sys
from pathlib import Path

def test_environment():
    """Test basic environment setup"""
    print("🧪 Azure Universal RAG - Environment Validation")
    print("=" * 50)
    
    # Test 1: Python version  
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Test 2: Project structure
    required_dirs = ["agents", "api", "services", "infrastructure", "config", "docs"]
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ Directory: {dir_name}")
        else:
            print(f"❌ Missing: {dir_name}")
    
    # Test 3: Key files
    key_files = [
        "requirements.txt",
        "pyproject.toml", 
        "scripts/setup_local_environment.py",
        "scripts/test_azure_connectivity.py",
        "docs/development/LOCAL_TESTING_IMPLEMENTATION_PLAN.md",
        "docs/getting-started/QUICK_START.md"
    ]
    
    for file_path in key_files:
        if Path(file_path).exists():
            print(f"✅ File: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Run: python scripts/setup_local_environment.py")
    print("2. Run: python scripts/test_azure_connectivity.py") 
    print("3. Follow: docs/getting-started/QUICK_START.md")
    
    print("\n📚 DOCUMENTATION:")
    print("- Local Testing Plan: docs/development/LOCAL_TESTING_IMPLEMENTATION_PLAN.md")
    print("- Quick Start Guide: docs/getting-started/QUICK_START.md")

if __name__ == "__main__":
    test_environment()
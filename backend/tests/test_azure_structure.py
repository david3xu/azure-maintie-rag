#!/usr/bin/env python3
"""
Test Azure Universal RAG Structure
Verifies all Azure integration files are properly created and importable
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

def test_azure_settings():
    """Test Azure settings configuration"""
    try:
        from config.azure_settings import azure_settings
        print("✅ Azure settings imported successfully")
        print(f"   Resource prefix: {azure_settings.azure_resource_prefix}")
        print(f"   Environment: {azure_settings.azure_environment}")
        return True
    except Exception as e:
        print(f"❌ Azure settings import failed: {e}")
        return False

def test_azure_structure():
    """Test Azure directory structure"""
    azure_dir = Path("backend/azure")
    required_files = [
        "__init__.py",
        "storage_client.py",
        "search_client.py",
        "cosmos_client.py",
        "ml_client.py"
    ]

    print("\n📁 Checking Azure directory structure...")
    all_exist = True

    for file_name in required_files:
        file_path = azure_dir / file_name
        if file_path.exists():
            print(f"   ✅ {file_name}")
        else:
            print(f"   ❌ {file_name} - MISSING")
            all_exist = False

    return all_exist

def test_infrastructure():
    """Test infrastructure files"""
    infra_dir = Path("infrastructure")
    required_files = [
        "azure-resources.bicep",
        "parameters.json",
        "provision.py"
    ]

    print("\n🏗️  Checking infrastructure files...")
    all_exist = True

    for file_name in required_files:
        file_path = infra_dir / file_name
        if file_path.exists():
            print(f"   ✅ {file_name}")
        else:
            print(f"   ❌ {file_name} - MISSING")
            all_exist = False

    return all_exist

def test_integrations():
    """Test integrations"""
    integrations_dir = Path("backend/integrations")
    required_files = [
        "azure_services.py"
    ]

    print("\n🔗 Checking integrations...")
    all_exist = True

    for file_name in required_files:
        file_path = integrations_dir / file_name
        if file_path.exists():
            print(f"   ✅ {file_name}")
        else:
            print(f"   ❌ {file_name} - MISSING")
            all_exist = False

    return all_exist

def main():
    """Main test function"""
    print("🧪 Azure Universal RAG Structure Test")
    print("=" * 50)

    # Test all components
    settings_ok = test_azure_settings()
    structure_ok = test_azure_structure()
    infra_ok = test_infrastructure()
    integrations_ok = test_integrations()

    # Summary
    print("\n📊 Test Summary:")
    print(f"   Azure Settings: {'✅ PASS' if settings_ok else '❌ FAIL'}")
    print(f"   Azure Structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"   Infrastructure: {'✅ PASS' if infra_ok else '❌ FAIL'}")
    print(f"   Integrations: {'✅ PASS' if integrations_ok else '❌ FAIL'}")

    all_passed = all([settings_ok, structure_ok, infra_ok, integrations_ok])

    if all_passed:
        print("\n🎉 All Azure Universal RAG components are properly structured!")
        print("📝 Next steps:")
        print("   1. Install Azure SDK packages: pip install azure-storage-blob azure-search-documents azure-cosmos azure-ai-ml azure-identity")
        print("   2. Configure Azure service credentials")
        print("   3. Test Azure service connections")
        print("   4. Deploy infrastructure using: python infrastructure/provision.py")
    else:
        print("\n⚠️  Some components need attention. Please check the missing files above.")

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
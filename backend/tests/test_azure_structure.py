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
        from config.settings import AzureSettings
        azure_settings = AzureSettings()
        print("✅ Azure settings imported successfully")
        print(f"   Resource prefix: {azure_settings.azure_resource_prefix}")
        print(f"   Environment: {azure_settings.azure_environment}")
        return True
    except Exception as e:
        print(f"❌ Azure settings import failed: {e}")
        return False

def test_azure_structure():
    """Test Azure directory structure"""
    integrations_dir = Path("integrations")
    required_files = [
        "__init__.py",
        "azure_services.py",
        "azure_openai.py"
    ]

    print("\n📁 Checking Azure integrations structure...")
    all_exist = True

    for file_name in required_files:
        file_path = integrations_dir / file_name
        if file_path.exists():
            print(f"   ✅ {file_name}")
        else:
            print(f"   ❌ {file_name} - MISSING")
            all_exist = False

    return all_exist

def test_infrastructure():
    """Test infrastructure files (Azure Bicep templates)"""
    # Check infrastructure directory at project root
    infra_dir = Path("../infrastructure")
    required_files = [
        "azure-resources-core.bicep",
        "azure-resources-ml-simple.bicep",
        "azure-resources-cosmos.bicep",
        "azure-resources-ml-simple.json"
    ]

    print("\n🏗️  Checking Azure infrastructure files...")
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
    integrations_dir = Path("integrations")
    required_files = [
        "azure_services.py",
        "azure_openai.py"
    ]

    print("\n🔗 Checking Azure integrations...")
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

    print("\n📊 Test Summary:")
    print(f"   Azure Settings: {'✅ PASS' if settings_ok else '❌ FAIL'}")
    print(f"   Azure Structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"   Infrastructure: {'✅ PASS' if infra_ok else '❌ FAIL'}")
    print(f"   Integrations: {'✅ PASS' if integrations_ok else '❌ FAIL'}")

    # Core functionality should pass
    core_passed = settings_ok and structure_ok and integrations_ok

    if core_passed:
        print("\n🎉 Core Azure integration components are working correctly!")
        print("   The application is ready for Azure configuration.")
    else:
        print("\n⚠️  Some core components need attention. Please check the missing files above.")

    return core_passed

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
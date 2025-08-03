#!/usr/bin/env python3
"""
Quick test to verify core features work after migration cleanup
Tests basic imports and structure without requiring Azure credentials
"""

import sys
from pathlib import Path
import traceback

def test_directory_structure():
    """Test that core directories exist in migrated structure"""
    print("🔍 Testing directory structure...")

    required_dirs = [
        'agents', 'api', 'services', 'infrastructure',
        'config', 'data', 'tests', 'scripts'
    ]

    missing_dirs = []
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)

    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False

    print(f"✅ All core directories present: {required_dirs}")
    return True


def test_basic_imports():
    """Test basic imports work with new structure"""
    print("🔍 Testing basic imports...")

    sys.path.insert(0, '.')

    import_tests = [
        ("agents.universal_agent", "Universal Agent"),
        ("api.main", "FastAPI App"),
        ("services.agent_service", "Agent Service"),
        ("config.settings", "Settings"),
    ]

    failed_imports = []

    for module_name, description in import_tests:
        try:
            __import__(module_name)
            print(f"✅ {description} ({module_name})")
        except Exception as e:
            print(f"❌ {description} ({module_name}): {str(e)}")
            failed_imports.append((module_name, str(e)))

    return len(failed_imports) == 0


def test_data_paths():
    """Test that data paths are correct after migration"""
    print("🔍 Testing data paths...")

    # Test that data directory exists
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print(f"❌ Data directory missing: {data_dir}")
        return False

    print(f"✅ Data directory exists: {data_dir}")

    # Check if Azure ML data file exists
    test_file = data_dir / "azure-ml" / "azure-machine-learning-azureml-api-2.md"
    if test_file.exists():
        size = test_file.stat().st_size
        print(f"✅ Test data file exists: {test_file} ({size:,} bytes)")
    else:
        print(f"⚠️  Test data file not found: {test_file} (optional)")

    return True


def main():
    """Run all migration validation tests"""
    print("🧪 Testing Azure Universal RAG after migration cleanup")
    print("=" * 60)

    tests = [
        ("Directory Structure", test_directory_structure),
        ("Basic Imports", test_basic_imports),
        ("Data Paths", test_data_paths),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")

    print(f"\nResult: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All migration validation tests passed!")
        print("✅ Directory structure migrated correctly")
        print("✅ Import paths updated successfully")
        print("✅ Data paths configured properly")
        return 0
    else:
        print("❌ Some validation tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

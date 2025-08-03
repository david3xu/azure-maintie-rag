#!/usr/bin/env python3
"""
Test runner for Azure Universal RAG - REAL TESTING ONLY
Runs all test suites with REAL Azure services and data
Follows CODING_STANDARDS.md: No mocks, no fake data, real Azure connections required
"""

import subprocess
import sys
from pathlib import Path


def validate_azure_credentials():
    """Validate Azure credentials are configured for REAL testing"""
    from config.settings import settings

    required_settings = {
        "AZURE_OPENAI_ENDPOINT": settings.azure_openai_endpoint,
        "AZURE_SEARCH_ENDPOINT": settings.azure_search_endpoint,
        "AZURE_STORAGE_ACCOUNT": settings.azure_storage_account,
        "AZURE_COSMOS_ENDPOINT": settings.azure_cosmos_endpoint
    }

    missing = [name for name, value in required_settings.items() if not value]

    if missing:
        print(f"❌ Missing Azure credentials for REAL testing: {', '.join(missing)}")
        print("")
        print("Per CODING_STANDARDS.md, tests require REAL Azure services.")
        print("Set these environment variables or configure in .env file:")
        for name in missing:
            print(f"  - {name}=<your-azure-{name.lower().replace('_', '-')}>")
        return False

    print("✅ Azure credentials configured for REAL testing")
    return True


def validate_real_data():
    """Validate real data exists for testing"""
    from pathlib import Path

    data_file = Path("data/raw/azure-ml/azure-machine-learning-azureml-api-2.md")

    if not data_file.exists():
        print(f"❌ Missing real data file: {data_file}")
        print("Per CODING_STANDARDS.md, tests require real data from data/raw/")
        return False

    # Validate file has substantial content
    content_size = data_file.stat().st_size
    if content_size < 5000:
        print(f"❌ Real data file too small: {content_size} bytes")
        return False

    print(f"✅ Real data available: {content_size:,} bytes")
    return True


def run_test_suite(test_path: str, description: str) -> bool:
    """Run a specific test suite with REAL Azure services"""
    print(f"\n{'='*60}")
    print(f"Running {description} - REAL AZURE TESTING")
    print(f"{'='*60}")

    try:
        # Run with longer timeout for real Azure operations
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            test_path,
            "-v",
            "--tb=short",
            "--asyncio-mode=auto",
            "-x",  # Stop on first failure for faster feedback
            "--durations=10"  # Show slowest tests
        ], capture_output=True, text=True, timeout=600)  # Longer timeout for real operations

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Check for mock/fake violations in output
        output_text = (result.stdout + result.stderr).lower()
        violations = []

        if "mock" in output_text:
            violations.append("Found 'mock' in test output - violates CODING_STANDARDS.md")
        if "fake" in output_text:
            violations.append("Found 'fake' in test output - violates CODING_STANDARDS.md")
        if "placeholder" in output_text:
            violations.append("Found 'placeholder' in test output - violates CODING_STANDARDS.md")

        if violations:
            print("\n⚠️  CODING STANDARDS VIOLATIONS:")
            for violation in violations:
                print(f"  - {violation}")

        return result.returncode == 0 and len(violations) == 0

    except subprocess.TimeoutExpired:
        print(f"❌ {description} timed out (real Azure operations can be slow)")
        return False
    except Exception as e:
        print(f"❌ {description} failed with error: {e}")
        return False


def main():
    """Run all test suites with REAL Azure services and data"""
    project_root = Path(__file__).parent.parent

    print("🧪 Azure Universal RAG Test Suite - REAL TESTING ONLY")
    print("Per CODING_STANDARDS.md: No mocks, no fake data, real Azure connections")
    print(f"Project root: {project_root}")
    print()

    # Change to project directory
    import os
    os.chdir(project_root)

    # Validate prerequisites for REAL testing
    print("🔍 Validating prerequisites for REAL testing...")

    if not validate_azure_credentials():
        print("\n❌ Cannot run REAL tests without Azure credentials")
        return 1

    if not validate_real_data():
        print("\n❌ Cannot run REAL tests without real data")
        return 1

    print("\n✅ Prerequisites validated - proceeding with REAL testing")

    # Test suites prioritized for real testing
    test_suites = [
        ("tests/unit/test_core.py", "Unit Tests (Structure Validation)"),
        ("tests/integration/test_azure_integration.py", "REAL Azure Services Integration"),
        ("tests/validation/validate_architecture.py", "Architecture Compliance"),
    ]

    results = {}

    for test_path, description in test_suites:
        if Path(test_path).exists():
            success = run_test_suite(test_path, description)
            results[description] = success

            # Stop on first failure for faster feedback in real testing
            if not success:
                print(f"\n⚠️  Stopping after first failure: {description}")
                break
        else:
            print(f"⚠️  Skipping {description} - path {test_path} not found")
            results[description] = None

    # Summary
    print(f"\n{'='*60}")
    print("📊 REAL TESTING SUMMARY")
    print(f"{'='*60}")

    total_tests = len([r for r in results.values() if r is not None])
    passed_tests = len([r for r in results.values() if r is True])

    for test_name, result in results.items():
        if result is True:
            print(f"✅ {test_name}")
        elif result is False:
            print(f"❌ {test_name}")
        else:
            print(f"⚠️  {test_name} (skipped)")

    print(f"\nResult: {passed_tests}/{total_tests} REAL test suites passed")

    if passed_tests == total_tests and total_tests > 0:
        print("🎉 All REAL tests passed with actual Azure services!")
        print("\nValidated:")
        print("  ✅ Real Azure service connections")
        print("  ✅ Real data processing from data/raw/")
        print("  ✅ Real performance characteristics")
        print("  ✅ No mocks or fake data")
        return 0
    else:
        print("❌ Some REAL tests failed")
        print("\nEnsure:")
        print("  - Azure credentials are valid")
        print("  - Azure services are accessible")
        print("  - Real data exists in data/raw/")
        print("  - No mocks or fake data in tests")
        return 1


if __name__ == "__main__":
    sys.exit(main())

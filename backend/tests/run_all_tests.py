#!/usr/bin/env python3
"""
Comprehensive test runner for the MaintIE Dual API Architecture

This script runs all tests to validate:
1. Dual API endpoints (multi-modal vs structured)
2. RAG architecture components
3. API functionality and performance
4. System integration and health

Usage:
    python backend/tests/run_all_tests.py
    python -m pytest backend/tests/ -v
"""

import subprocess
import sys
import time
from pathlib import Path

def run_test_file(test_file: str, description: str):
    """Run a specific test file and report results"""
    print(f"\n{'='*60}")
    print(f"🧪 Running {description}")
    print(f"📁 File: {test_file}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Run the test file
        result = subprocess.run([
            sys.executable, test_file
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout

        end_time = time.time()
        duration = end_time - start_time

        if result.returncode == 0:
            print(f"✅ {description} - PASSED ({duration:.2f}s)")
            if result.stdout:
                print("📋 Output:")
                print(result.stdout)
        else:
            print(f"❌ {description} - FAILED ({duration:.2f}s)")
            if result.stderr:
                print("🚨 Errors:")
                print(result.stderr)
            if result.stdout:
                print("📋 Output:")
                print(result.stdout)

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT (>5 minutes)")
        return False
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False

def run_pytest_tests():
    """Run pytest on all test files"""
    print(f"\n{'='*60}")
    print("🧪 Running Pytest Test Suite")
    print(f"{'='*60}")

    try:
        # Run pytest on the tests directory
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout

        if result.returncode == 0:
            print("✅ Pytest suite - PASSED")
        else:
            print("❌ Pytest suite - FAILED")

        if result.stdout:
            print("📋 Output:")
            print(result.stdout)

        if result.stderr:
            print("🚨 Errors:")
            print(result.stderr)

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("⏰ Pytest suite - TIMEOUT (>10 minutes)")
        return False
    except Exception as e:
        print(f"💥 Pytest suite - ERROR: {e}")
        return False

def check_api_health():
    """Check if the API is running and healthy"""
    print(f"\n{'='*60}")
    print("🏥 Checking API Health")
    print(f"{'='*60}")

    try:
        import requests
    except ImportError:
        print("❌ API Health: requests module not available")
        return False

    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/api/v1/health", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ API Health: Status {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ API Health: Connection failed - Is the API server running?")
        return False
    except Exception as e:
        print(f"💥 API Health: Error - {e}")
        return False

def main():
    """Main test runner function"""
    print("🚀 MaintIE Dual API Architecture - Comprehensive Test Suite")
    print("Testing the complete dual API implementation")

        # Change to the backend directory
    backend_dir = Path(__file__).parent.parent
    original_dir = Path.cwd()

    try:
        # Change to backend directory
        import os
        os.chdir(backend_dir)

        # Test results tracking
        test_results = []

        # Check API health first
        api_healthy = check_api_health()
        if not api_healthy:
            print("\n⚠️  API is not running. Some tests may fail.")
            print("   Start the API with: python -m uvicorn api.main:app --reload")

        # Run individual test files
        test_files = [
            ("tests/test_dual_api.py", "Dual API Endpoint Tests"),
            ("tests/test_rag_architecture.py", "RAG Architecture Tests"),
            ("tests/test_real_api.py", "Real API Integration Tests"),
            ("tests/test_real_pipeline.py", "Real Pipeline Tests"),
        ]

        for test_file, description in test_files:
            if Path(test_file).exists():
                success = run_test_file(test_file, description)
                test_results.append((description, success))
            else:
                print(f"⚠️  Test file not found: {test_file}")
                test_results.append((description, False))

        # Run pytest suite
        pytest_success = run_pytest_tests()
        test_results.append(("Pytest Suite", pytest_success))

        # Summary
        print(f"\n{'='*60}")
        print("📊 TEST SUMMARY")
        print(f"{'='*60}")

        passed = 0
        failed = 0

        for description, success in test_results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{status} - {description}")
            if success:
                passed += 1
            else:
                failed += 1

        print(f"\n📈 Results: {passed} passed, {failed} failed")

        if failed == 0:
            print("🎉 All tests passed! The dual API architecture is working correctly.")
        else:
            print("⚠️  Some tests failed. Please check the output above for details.")

        print(f"\n🔗 API Documentation: http://localhost:8000/docs")
        print(f"🔗 Health Check: http://localhost:8000/api/v1/health")

        return failed == 0

    finally:
        # Change back to original directory
        import os
        os.chdir(original_dir)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
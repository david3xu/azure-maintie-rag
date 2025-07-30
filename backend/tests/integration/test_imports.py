#!/usr/bin/env python3
"""Test imports and dependencies for modified Azure client files"""

import sys
import os
import traceback

def test_file_imports(filepath, module_name):
    """Test that a file can be imported without errors"""
    try:
        # Add the backend directory to Python path
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)
        
        # Import the module
        __import__(module_name)
        print(f"✅ {filepath}: Import successful")
        return True
    except ImportError as e:
        print(f"❌ {filepath}: Import error: {e}")
        print(f"   Full traceback:\n{traceback.format_exc()}")
        return False
    except Exception as e:
        print(f"❌ {filepath}: Unexpected error during import: {e}")
        print(f"   Full traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    files_to_test = [
        ("core/azure_storage/storage_client.py", "core.azure_storage.storage_client"),
        ("core/azure_search/search_client.py", "core.azure_search.search_client"),
        ("core/azure_cosmos/cosmos_gremlin_client.py", "core.azure_cosmos.cosmos_gremlin_client"),
        ("core/azure_ml/ml_client.py", "core.azure_ml.ml_client"),
        ("core/azure_ml/gnn/feature_engineering.py", "core.azure_ml.gnn.feature_engineering")
    ]
    
    print("Testing imports for modified files...")
    all_passed = True
    
    for file_path, module_name in files_to_test:
        if not test_file_imports(file_path, module_name):
            all_passed = False
        print()  # Add spacing between tests
    
    if all_passed:
        print("✅ All files imported successfully")
        sys.exit(0)
    else:
        print("❌ Some files failed to import")
        sys.exit(1)
#!/usr/bin/env python3
"""
Directory structure validation for Azure Universal RAG
Prevents common directory structure issues and enforces architecture compliance
"""

import os
import sys
from pathlib import Path

def validate_directory_structure():
    """Validate project directory structure"""
    # Ensure we're in the project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    issues = []
    
    # Check for required top-level directories
    required_dirs = ['backend', 'frontend', 'infra', 'docs']
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            issues.append(f"❌ Missing required directory: {dir_name}")
    
    # Check backend structure
    backend_dirs = ['api', 'agents', 'infra', 'services', 'tests']
    for dir_name in backend_dirs:
        backend_path = Path('backend') / dir_name
        if not backend_path.exists():
            issues.append(f"❌ Missing backend directory: {backend_path}")
    
    # Check for problematic directories that shouldn't exist
    problematic_dirs = [
        './venv',
        './__pycache__',
        'frontend/node_modules',
        './data/outputs',
        './scripts/azure_ml/mlruns'
    ]
    
    for dir_path in problematic_dirs:
        if Path(dir_path).exists():
            issues.append(f"⚠️  Temporary directory should be cleaned: {dir_path}")
    
    # Check for large files that shouldn't be committed
    large_extensions = ['.pth', '.pkl', '.h5', '.onnx', '.bin']
    for root, dirs, files in os.walk('.'):
        # Skip .git and other hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if any(file.endswith(ext) for ext in large_extensions):
                file_path = Path(root) / file
                if file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB
                    issues.append(f"⚠️  Large file detected: {file_path} ({file_path.stat().st_size / 1024 / 1024:.1f}MB)")
    
    # Check for proper .gitignore patterns
    gitignore_path = Path('.gitignore')
    if gitignore_path.exists():
        gitignore_content = gitignore_path.read_text()
        required_patterns = ['venv/', '__pycache__/', '*.pyc', 'node_modules/', '*.log']
        for pattern in required_patterns:
            if pattern not in gitignore_content:
                issues.append(f"⚠️  Missing .gitignore pattern: {pattern}")
    
    # Report results
    if issues:
        print("🔍 Directory Structure Validation Issues:")
        for issue in issues:
            print(f"  {issue}")
        return len([i for i in issues if i.startswith('❌')])  # Only fail on errors, not warnings
    else:
        print("✅ Directory structure validation passed")
        return 0

if __name__ == "__main__":
    sys.exit(validate_directory_structure())
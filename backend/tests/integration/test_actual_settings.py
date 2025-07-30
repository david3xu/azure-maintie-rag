#!/usr/bin/env python3
"""
Test Actual Settings Loading
"""

import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

print("🔍 Actual Settings Test")
print("=" * 40)

# Check OS environment first
print(f"OS environment COSMOS_CONTAINER_NAME: {repr(os.getenv('COSMOS_CONTAINER_NAME'))}")

# Import fresh settings
import importlib
if 'config.settings' in sys.modules:
    importlib.reload(sys.modules['config.settings'])

from config.settings import Settings

# Create new instance
fresh_settings = Settings()
print(f"Fresh settings container: {repr(fresh_settings.azure_cosmos_container)}")

# Check if environment is being loaded
print(f"Fresh settings dict cosmos container: {repr(fresh_settings.model_dump().get('azure_cosmos_container'))}")

# Check field definition
field_info = fresh_settings.model_fields.get('azure_cosmos_container')
print(f"Field info: {field_info}")
if field_info:
    print(f"Field env name: {getattr(field_info, 'alias', None) or getattr(field_info, 'serialization_alias', None)}")
    
# Test direct environment loading
print(f"Config env_file: {fresh_settings.model_config.get('env_file')}")
print(f"Config extra: {fresh_settings.model_config.get('extra')}")
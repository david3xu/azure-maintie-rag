#!/usr/bin/env python3
"""
Debug Environment Loading
"""

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

print("🔍 Environment Loading Debug")
print("=" * 40)

# Check OS environment
print(f"OS environment COSMOS_CONTAINER_NAME: {repr(os.getenv('COSMOS_CONTAINER_NAME'))}")

# Check .env file
env_file = Path(".env")
print(f".env file exists: {env_file.exists()}")
if env_file.exists():
    content = env_file.read_text()
    cosmos_lines = [line for line in content.split('\n') if 'COSMOS_CONTAINER' in line]
    print(f".env COSMOS_CONTAINER_NAME lines: {cosmos_lines}")

# Test simple settings class
class TestSettings(BaseSettings):
    cosmos_container_name: str = Field(default="", env="COSMOS_CONTAINER_NAME")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

test_settings = TestSettings()
print(f"Test settings container: {repr(test_settings.cosmos_container_name)}")

# Test with _env_file parameter
test_settings2 = TestSettings(_env_file=".env")
print(f"Test settings2 container: {repr(test_settings2.cosmos_container_name)}")
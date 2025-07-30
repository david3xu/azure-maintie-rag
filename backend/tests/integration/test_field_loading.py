#!/usr/bin/env python3
"""
Test Field Loading Approaches
"""

import os
from pydantic import Field
from pydantic_settings import BaseSettings

# Set environment variable
os.environ['COSMOS_CONTAINER_NAME'] = 'knowledge-graph-staging'

print("🔍 Field Loading Test")
print("=" * 40)

# Method 1: Field with env parameter
class Method1Settings(BaseSettings):
    azure_cosmos_container: str = Field(default="", env="COSMOS_CONTAINER_NAME")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings1 = Method1Settings()
print(f"Method 1 (Field env): {repr(settings1.azure_cosmos_container)}")

# Method 2: Direct environment field name matching  
class Method2Settings(BaseSettings):
    cosmos_container_name: str = Field(default="")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings2 = Method2Settings()
print(f"Method 2 (direct name): {repr(settings2.cosmos_container_name)}")

# Method 3: Using model_config
class Method3Settings(BaseSettings):
    azure_cosmos_container: str = Field(default="")
    
    model_config = {
        "env_file": ".env",
        "extra": "ignore",
        "env_prefix": "",
    }

settings3 = Method3Settings()
print(f"Method 3 (model_config): {repr(settings3.azure_cosmos_container)}")

# Check versions
import pydantic
import pydantic_settings
print(f"\nVersions:")
print(f"Pydantic: {pydantic.version.VERSION}")
print(f"Pydantic-settings: {pydantic_settings.__version__}")
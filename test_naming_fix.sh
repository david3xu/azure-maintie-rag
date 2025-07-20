#!/bin/bash

# Test script to verify the naming service fix
set -euo pipefail

# Source the naming service
source scripts/azure-naming-service.sh

echo "Testing naming service fix..."

# Test storage name generation
echo "Testing storage name generation:"
storage_name=$(generate_globally_unique_storage_name "maintie" "dev")
echo "Captured storage name: '$storage_name'"
echo "Length: ${#storage_name}"

# Test search name generation
echo "Testing search name generation:"
search_name=$(generate_unique_search_name "maintie" "dev" "eastus")
echo "Captured search name: '$search_name'"
echo "Length: ${#search_name}"

# Test key vault name generation
echo "Testing key vault name generation:"
keyvault_name=$(generate_unique_keyvault_name "maintie" "dev")
echo "Captured key vault name: '$keyvault_name'"
echo "Length: ${#keyvault_name}"

echo "Test completed successfully!"
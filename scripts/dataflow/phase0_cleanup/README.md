# Phase 0: System Cleanup

**Purpose**: Ensure completely clean state for unbiased testing

## Scripts in This Phase:

- 00_01_cleanup_all_services.py - Clean all Azure services data
- 00_02_cleanup_azure_storage.py - Clean Azure Storage blobs
- 00_03_verify_clean_state.py - Verify clean state achieved

## Execution Order:

Run scripts in numerical order (01, 02, 03, ...).

## Prerequisites:

- Previous phase must be completed successfully
- Azure services must be operational
- No lock files from previous executions

## Dependencies:

Each script checks prerequisites before execution and creates completion markers.

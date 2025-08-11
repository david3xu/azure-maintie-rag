# Phase 2: Data Ingestion

**Purpose**: Process real data through complete ingestion pipeline

## Scripts in This Phase:

- 02_02_storage_upload_primary.py - Primary data upload pipeline
- 02_02_storage_upload_direct.py - Direct Azure Storage operations
- 02_03_vector_embeddings.py - Generate vector embeddings
- 02_04_search_indexing.py - Index documents in Cognitive Search

## Execution Order:

Run scripts in numerical order (01, 02, 03, ...).

## Prerequisites:

- Previous phase must be completed successfully
- Azure services must be operational
- No lock files from previous executions

## Dependencies:

Each script checks prerequisites before execution and creates completion markers.

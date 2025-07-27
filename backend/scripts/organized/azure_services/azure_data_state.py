import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from integrations.azure_services import AzureServicesManager

async def main():
    services = AzureServicesManager()
    state = await services.validate_domain_data_state('general')
    print('Azure Data State Analysis:')
    print(f'  Blob Storage: {state["azure_blob_storage"]["document_count"]} documents')
    print(f'  Search Index: {state["azure_cognitive_search"]["document_count"]} documents')
    print(f'  Cosmos DB: {state["azure_cosmos_db"]["vertex_count"]} entities')
    print(f'  Raw Data: {state["raw_data_directory"]["file_count"]} files')
    print(f'  Processing Required: {state["requires_processing"]}')

if __name__ == '__main__':
    asyncio.run(main())
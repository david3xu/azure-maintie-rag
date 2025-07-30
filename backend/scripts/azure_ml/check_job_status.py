#!/usr/bin/env python3
"""
Check Azure ML job status
"""
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import sys

def check_job_status(job_id):
    """Check the status of an Azure ML job"""
    subscription_id = "ccc6af52-5928-4dbe-8ceb-fa794974a30f"
    resource_group = "rg-maintie-rag-staging"
    workspace_name = "ml-maintierag-lnpxxab4"
    
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )
    
    # Extract job name from full ID
    job_name = job_id.split('/')[-1]
    print(f"Checking job: {job_name}")
    
    try:
        job = ml_client.jobs.get(job_name)
        print(f"Status: {job.status}")
        print(f"Creation Time: {job.creation_context.created_at}")
        
        if hasattr(job, 'log_files'):
            print(f"Log files available: {job.log_files}")
        
        return job
    except Exception as e:
        print(f"Error checking job: {e}")
        return None

if __name__ == "__main__":
    job_id = "standalone-gnn-training-1753852905"  # Latest job
    if len(sys.argv) > 1:
        job_id = sys.argv[1]
    
    check_job_status(job_id)
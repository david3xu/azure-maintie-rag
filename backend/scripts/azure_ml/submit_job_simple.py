#!/usr/bin/env python3
"""
Simple Azure ML job submission using environment variables directly
"""
import os
import time
from azure.ai.ml import MLClient, command, Output
from azure.identity import DefaultAzureCredential

def submit_gnn_training_job():
    """Submit GNN training job to Azure ML"""
    print("🚀 Submitting GNN Training Job to Azure ML...")
    
    # Get configuration from environment variables
    subscription_id = "ccc6af52-5928-4dbe-8ceb-fa794974a30f"  # From staging.env
    resource_group = "rg-maintie-rag-staging"  # From staging.env
    workspace_name = "ml-maintierag-lnpxxab4"  # From staging.env
    
    print(f"📋 Configuration:")
    print(f"   Subscription: {subscription_id}")
    print(f"   Resource Group: {resource_group}")
    print(f"   Workspace: {workspace_name}")
    
    try:
        # Initialize Azure ML client
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        print("✅ Azure ML client initialized")
        
        # List available compute targets
        print("💻 Checking available compute targets...")
        compute_targets = list(ml_client.compute.list())
        
        if not compute_targets:
            print("❌ No compute targets found!")
            print("📝 You need to create a compute cluster in Azure ML Studio first")
            return None
            
        # Show available compute targets
        print(f"🎯 Available compute targets:")
        for compute in compute_targets:
            print(f"   - {compute.name} ({compute.type}) - {compute.provisioning_state}")
        
        # Use first available compute target
        compute_name = compute_targets[0].name
        print(f"🎯 Selected compute: {compute_name}")
        
        # Create unique job name
        job_name = f"gnn-real-data-training-{int(time.time())}"
        
        # Create training job
        job = command(
            name=job_name,
            display_name="GNN Training - Real Cosmos DB Data",
            description="Train GNN model using real Azure Cosmos DB maintenance data",
            code="./",  # Current directory with training script
            command="python train_gnn_pytorch_only.py --epochs 100 --lr 0.001 --hidden_dim 256",
            environment="azureml:AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu@latest",
            compute=compute_name,
            outputs={
                "model_output": Output(type="uri_folder", mode="rw_mount")
            },
            experiment_name="real-gnn-maintenance-training"
        )
        
        print("📤 Submitting job to Azure ML...")
        submitted_job = ml_client.jobs.create_or_update(job)
        
        print("🎉 Job submitted successfully!")
        print(f"🆔 Job ID: {submitted_job.id}")
        print(f"📊 Job Name: {submitted_job.display_name}")
        print(f"🔗 Status: {submitted_job.status}")
        print(f"🌐 Monitor at: https://ml.azure.com/runs/{submitted_job.id}")
        
        # Wait a bit and check initial status
        print("⏳ Checking initial job status...")
        time.sleep(5)
        
        try:
            current_job = ml_client.jobs.get(submitted_job.id)
            print(f"📊 Current Status: {current_job.status}")
        except Exception as e:
            print(f"⚠️ Could not check status: {e}")
        
        return {
            "job_id": submitted_job.id,
            "studio_url": f"https://ml.azure.com/runs/{submitted_job.id}",
            "compute_used": compute_name
        }
        
    except Exception as e:
        print(f"❌ Failed to submit job: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Azure ML GNN Training Job Submission")
    print("=" * 50)
    
    result = submit_gnn_training_job()
    
    if result:
        print("\n" + "=" * 50)
        print("🏆 SUCCESS: GNN Training Job Submitted!")
        print("=" * 50)
        print(f"🆔 Job ID: {result['job_id']}")
        print(f"🌐 Monitor: {result['studio_url']}")
        print(f"💻 Compute: {result['compute_used']}")
        print("\n💡 The job will train GNN with REAL Azure Cosmos DB data!")
        print("🔍 Check Azure ML Studio to monitor progress")
    else:
        print("\n❌ Failed to submit job")
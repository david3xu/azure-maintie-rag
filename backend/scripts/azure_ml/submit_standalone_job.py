#!/usr/bin/env python3
"""
Submit standalone GNN training job to Azure ML - addresses local vs cloud gap
This script uses the train_gnn_standalone.py which includes all dependencies inline
"""
import os
import time
from azure.ai.ml import MLClient, command, Output
from azure.identity import DefaultAzureCredential

def submit_standalone_gnn_job():
    """Submit standalone GNN training job to Azure ML"""
    print("🚀 Submitting STANDALONE GNN Training Job to Azure ML...")
    print("🔧 This version has ALL dependencies inline - no external 'core' module dependencies")
    print("=" * 60)
    
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
        job_name = f"standalone-gnn-training-{int(time.time())}"
        
        # Create training job using STANDALONE script
        job = command(
            name=job_name,
            display_name="Standalone GNN Training - Real Cosmos DB Data (No External Dependencies)",
            description="Train GNN using standalone script with ALL dependencies inline - fixes local vs cloud gap",
            code="./",  # Current directory with training script
            command="python train_gnn_standalone.py --epochs 100 --lr 0.001 --hidden_dim 256 --output_dir ${{outputs.model_output}}",
            environment="azureml:AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu@latest",
            compute=compute_name,
            outputs={
                "model_output": Output(type="uri_folder", mode="rw_mount")
            },
            experiment_name="standalone-gnn-real-data-training"
        )
        
        print("📤 Submitting STANDALONE job to Azure ML...")
        print("🔧 This job should use REAL data in cloud environment!")
        submitted_job = ml_client.jobs.create_or_update(job)
        
        print("🎉 STANDALONE Job submitted successfully!")
        print(f"🆔 Job ID: {submitted_job.id}")
        print(f"📊 Job Name: {submitted_job.display_name}")
        print(f"🔗 Status: {submitted_job.status}")
        print(f"🌐 Monitor at: https://ml.azure.com/runs/{submitted_job.id}")
        
        # Monitor initial status
        print("⏳ Checking initial job status...")
        for i in range(6):  # Check for 3 minutes
            try:
                time.sleep(30)
                current_job = ml_client.jobs.get(submitted_job.id)
                status = current_job.status
                print(f"   [{i+1}/6] Status: {status}")
                
                if status in ["Completed", "Failed", "Canceled"]:
                    break
                    
            except Exception as e:
                print(f"   Error checking status: {e}")
                break
        
        print(f"\n" + "=" * 60)
        print("🎯 STANDALONE GNN TRAINING JOB SUMMARY:")
        print("=" * 60)
        print(f"🆔 Job ID: {submitted_job.id}")
        print(f"💻 Compute: {compute_name}")
        print(f"📁 Script: train_gnn_standalone.py (ALL dependencies inline)")
        print(f"🌐 Monitor: https://ml.azure.com/runs/{submitted_job.id}")
        print(f"🔧 Key Feature: NO external 'core' module dependencies")
        print(f"📊 Expected Data: REAL Azure Cosmos DB data (not synthetic)")
        
        return {
            "job_id": submitted_job.id,
            "studio_url": f"https://ml.azure.com/runs/{submitted_job.id}",
            "compute_used": compute_name,
            "script_used": "train_gnn_standalone.py",
            "has_inline_dependencies": True
        }
        
    except Exception as e:
        print(f"❌ Failed to submit standalone job: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Azure ML STANDALONE GNN Training Job Submission")
    print("=" * 60)
    
    result = submit_standalone_gnn_job()
    
    if result:
        print("\n" + "=" * 60)
        print("🏆 SUCCESS: STANDALONE GNN Training Job Submitted!")
        print("=" * 60)
        print(f"🆔 Job ID: {result['job_id']}")
        print(f"🌐 Monitor: {result['studio_url']}")
        print(f"💻 Compute: {result['compute_used']}")
        print(f"📁 Script: {result['script_used']}")
        print(f"🔧 Inline Dependencies: {result['has_inline_dependencies']}")
        print("\n💡 This job should now use REAL Cosmos DB data in Azure ML cloud!")
        print("🔍 Check Azure ML Studio to verify it's NOT using synthetic data")
    else:
        print("\n❌ Failed to submit standalone job")
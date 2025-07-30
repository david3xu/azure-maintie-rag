#!/usr/bin/env python3
"""
Submit REAL Azure ML GNN training job
"""
import sys
import os
import time

# Fix Python path for Azure ML submission
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
    
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def submit_real_azure_ml_job():
    """Submit actual Azure ML training job"""
    print("🚀 Submitting REAL Azure ML GNN Training Job...")
    print("=" * 60)
    
    try:
        from azure.ai.ml import MLClient, command, Input, Output
        from core.azure_ml.client import AzureMLClient
        from config.settings import azure_settings
        
        # Initialize ML client
        ml_wrapper = AzureMLClient()
        ml_client = ml_wrapper.ml_client
        
        print("✅ Azure ML client initialized")
        print(f"🏢 Workspace: {azure_settings.azure_ml_workspace_name}")
        
        # List compute targets
        compute_targets = list(ml_client.compute.list())
        print(f"💻 Available compute targets:")
        for compute in compute_targets:
            print(f"   - {compute.name} ({compute.type}) - {compute.provisioning_state}")
        
        if not compute_targets:
            print("❌ No compute targets available!")
            return None
            
        # Use first available compute
        compute_name = compute_targets[0].name
        print(f"🎯 Selected compute: {compute_name}")
        
        # Create job
        job_name = f"real-gnn-training-{int(time.time())}"
        
        job = command(
            name=job_name,
            display_name="Real GNN Training - Maintenance Domain",
            description="Real Azure ML GNN training using Cosmos DB data",
            code="./",  # Current directory
            command="python train_gnn_pytorch_only.py --epochs 100 --lr 0.001 --hidden_dim 256 --output_dir ${{outputs.model_output}}",
            environment="azureml:AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu@latest",
            compute=compute_name,
            outputs={
                "model_output": Output(type="uri_folder")
            },
            experiment_name="real-gnn-maintenance-training"
        )
        
        print("📤 Submitting job to Azure ML...")
        submitted_job = ml_client.jobs.create_or_update(job)
        
        print("✅ REAL Azure ML job submitted!")
        print(f"🆔 Job ID: {submitted_job.id}")
        print(f"📊 Job Name: {submitted_job.display_name}")
        print(f"🔗 Status: {submitted_job.status}")
        print(f"💻 Compute: {compute_name}")
        print(f"🧪 Experiment: real-gnn-maintenance-training")
        
        # Monitor job for first few minutes
        print(f"\n🔄 Monitoring job status...")
        for i in range(12):  # Check for 6 minutes
            try:
                current_job = ml_client.jobs.get(submitted_job.id)
                status = current_job.status
                print(f"   [{i+1}/12] Status: {status}")
                
                if status in ["Completed", "Failed", "Canceled"]:
                    break
                    
                time.sleep(30)  # Wait 30 seconds
            except Exception as e:
                print(f"   Error checking status: {e}")
                break
        
        # Get final status
        try:
            final_job = ml_client.jobs.get(submitted_job.id)
            final_status = final_job.status
        except:
            final_status = "Unknown"
        
        print(f"\n" + "=" * 60)
        print("🎯 REAL AZURE ML JOB SUBMITTED:")
        print("=" * 60)
        print(f"🆔 Job ID: {submitted_job.id}")
        print(f"📊 Current Status: {final_status}")
        print(f"💻 Compute: {compute_name}")
        print(f"📁 Model Output: azureml://datastores/workspaceblobstore/paths/gnn-models/{job_name}/")
        print(f"🌐 Azure ML Studio: https://ml.azure.com/runs/{submitted_job.id}")
        
        print(f"\n💡 Job Status:")
        if final_status == "Completed":
            print("🎉 SUCCESS: Training completed!")
        elif final_status == "Running":
            print("⏳ Job is still running - check Azure ML Studio for progress")
        elif final_status == "Failed":
            print("❌ Job failed - check Azure ML Studio for error details")
        else:
            print(f"ℹ️  Status: {final_status}")
        
        return {
            "job_id": submitted_job.id,
            "status": final_status,
            "compute_used": compute_name,
            "real_azure_ml": True,
            "studio_url": f"https://ml.azure.com/runs/{submitted_job.id}"
        }
        
    except Exception as e:
        print(f"❌ Failed to submit Azure ML job: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main execution"""
    print("REAL Azure ML GNN Training Job Submission")
    print("=" * 60)
    
    result = submit_real_azure_ml_job()
    
    if result:
        print(f"\n🏆 SUCCESS: Real Azure ML job submitted!")
        print(f"🆔 Job ID: {result['job_id']}")
        print(f"🌐 Monitor at: {result['studio_url']}")
        return 0
    else:
        print(f"\n💥 FAILURE: Could not submit Azure ML job")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
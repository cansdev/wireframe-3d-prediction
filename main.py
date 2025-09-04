import torch
from demo_dataset.PCtoWFdataset import PCtoWFdataset
from train import evaluate_model, train_overfit_model
from test import compute_building3d_metrics
import wandb

def main():
    # Initialize W&B run
    run = wandb.init(
        entity="can_g-a",
        project="Wireframe3D",
        config={
            "learning_rate": 0.001,
            "architecture": "PointCloudToWireframe",
            "dataset": "PCtoWFdataset (demo_dataset train/test)",
            "epochs": 1000,
        },
    )
    
    # Load data
    dataset = PCtoWFdataset( 
        train_pc_dir='demo_dataset/train_dataset/point_cloud',
        train_wf_dir='demo_dataset/train_dataset/wireframe',
        test_pc_dir='demo_dataset/test_dataset/point_cloud',
        test_wf_dir='demo_dataset/test_dataset/wireframe'
    )
    
    # Print dataset info
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*60)
    print("STARTING BATCH TRAINING")
    print("="*60)

    # Load training dataset with multiple files
    # Initializes sample object
    train_dataset = dataset.load_training_dataset()
    
    # Load and preprocess all training data at once
    # Creates sample object, loads data, creates adjacency matrix, normalizes data
    train_dataset.load_all_data()

    # Gets batch data
    batch_data = train_dataset.get_batch_data(target_points=1024)

    model, loss_history = train_overfit_model(batch_data, num_epochs=1000, learning_rate=0.001, wandb_run=run)

    print("\n" + "="*50)
    print("EVALUATING TRAINED MODEL")
    print("="*50)
    
    # Get max vertices for evaluation
    max_vertices = train_dataset.max_vertices
    
    # Load and evaluate test dataset
    test_dataset = dataset.load_testing_dataset()
    test_dataset.load_all_data()
    test_batch_data = test_dataset.get_batch_data(target_points=1024)
    
    test_results = evaluate_model(model, test_batch_data, device, max_vertices)
    
    # Compute Building3D metrics
    building3d_metrics = compute_building3d_metrics(test_results)
    
    # Log Building3D metrics to W&B
    run.log({
        "eval_building3d_aco": building3d_metrics['building3d_aco'],
        "eval_building3d_cp": building3d_metrics['building3d_cp'],
        "eval_building3d_cr": building3d_metrics['building3d_cr'],
        "eval_building3d_c_f1": building3d_metrics['building3d_c_f1'],
        "eval_building3d_ep": building3d_metrics['building3d_ep'],
        "eval_building3d_er": building3d_metrics['building3d_er'],
        "eval_building3d_e_f1": building3d_metrics['building3d_e_f1'],
    })
    
    # Also log with simpler names for better visibility
    run.log({
        "Building3D_ACO": building3d_metrics['building3d_aco'],
        "Building3D_CP": building3d_metrics['building3d_cp'],
        "Building3D_CR": building3d_metrics['building3d_cr'],
        "Building3D_C_F1": building3d_metrics['building3d_c_f1'],
        "Building3D_EP": building3d_metrics['building3d_ep'],
        "Building3D_ER": building3d_metrics['building3d_er'],
        "Building3D_E_F1": building3d_metrics['building3d_e_f1'],
    })
    
    # Log a summary metric for easy comparison
    run.log({
        "Building3D_Overall_Score": (building3d_metrics['building3d_c_f1'] + building3d_metrics['building3d_e_f1']) / 2
    })
    
    # Print results
    print("\nTest Results:")
    print("-" * 40)
    print("Building3D Benchmark Metrics:")
    print(f"  ACO: {building3d_metrics['building3d_aco']:.6f}")
    print(f"  CP: {building3d_metrics['building3d_cp']:.6f}")
    print(f"  CR: {building3d_metrics['building3d_cr']:.6f}")
    print(f"  C-F1: {building3d_metrics['building3d_c_f1']:.6f}")
    print(f"  EP: {building3d_metrics['building3d_ep']:.6f}")
    print(f"  ER: {building3d_metrics['building3d_er']:.6f}")
    print(f"  E-F1: {building3d_metrics['building3d_e_f1']:.6f}")
    print()
    
    print("Custom Metrics:")
    for result in test_results:
        print(f"Sample {result['sample_index']+1}:")
        print(f"  Vertex RMSE: {result['vertex_rmse']:.6f}")
        print(f"  Edge Accuracy: {result['edge_accuracy']:.6f}")
        print(f"  Edge Precision: {result['edge_precision']:.6f}")
        print(f"  Edge Recall: {result['edge_recall']:.6f}")
        print(f"  Edge F1-Score: {result['edge_f1_score']:.6f}")
        print()
    
    # Save model
    torch.save(model.state_dict(), 'trained_model.pth')
    print("\nModel saved as 'trained_model.pth'")
    
    # Log summary report as W&B artifact
    import os
    report_path = 'output/summary_report.txt'
    if os.path.exists(report_path):
        artifact = wandb.Artifact(
            name="evaluation_summary_report",
            type="evaluation_report",
            description="Comprehensive evaluation results with Building3D and custom metrics"
        )
        artifact.add_file(report_path)
        run.log_artifact(artifact)
        print(f"✓ Summary report logged to W&B as artifact: {artifact.name}")
    else:
        print("⚠ Warning: summary_report.txt not found, skipping W&B artifact logging")
    
    # Save W&B run ID for later use by evaluate.py
    run_id = run.id
    with open('wandb_run_id.txt', 'w') as f:
        f.write(run_id)
    print(f"✓ W&B run ID saved: {run_id}")
    
    # Finish the W&B run
    run.finish()
    
    print("\nOvertraining completed successfully!")

if __name__ == "__main__":
    main()

import torch
from demo_dataset.PCtoWFdataset import PCtoWFdataset
from train import train_overfit_model
from evaluate import analyze_individual_predictions
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
    
    # Save the trained model first
    torch.save(model.state_dict(), 'trained_model.pth')
    
    # Load and evaluate test dataset using comprehensive evaluation
    test_dataset = dataset.load_testing_dataset()
    test_dataset.load_all_data()
    
    # Evaluate each test sample individually using comprehensive metrics
    test_results = []
    for i, individual_sample in enumerate(test_dataset.samples):
        sample_name = f"test_sample_{i+1}"
        analysis = analyze_individual_predictions(model, individual_sample, device, sample_name)
        test_results.append({
            'sample_index': i,
            'vertex_rmse': analysis['vertex_rmse'],
            'edge_accuracy': analysis['edge_precision'],  # Use precision as accuracy proxy
            'edge_precision': analysis['edge_precision'],
            'edge_recall': analysis['edge_recall'],
            'edge_f1_score': analysis['edge_f1_score'],
            'building3d_metrics': {
                'building3d_aco': analysis['building3d_aco'],
                'building3d_cp': analysis['building3d_cp'],
                'building3d_cr': analysis['building3d_cr'],
                'building3d_c_f1': analysis['building3d_c_f1'],
                'building3d_ep': analysis['building3d_ep'],
                'building3d_er': analysis['building3d_er'],
                'building3d_e_f1': analysis['building3d_e_f1']
            }
        })
    
    # Calculate aggregate metrics
    avg_vertex_rmse = sum(r['vertex_rmse'] for r in test_results) / len(test_results)
    avg_edge_precision = sum(r['edge_precision'] for r in test_results) / len(test_results)
    avg_edge_recall = sum(r['edge_recall'] for r in test_results) / len(test_results)
    avg_edge_f1 = sum(r['edge_f1_score'] for r in test_results) / len(test_results)
    
    # Building3D metrics averages
    avg_building3d_aco = sum(r['building3d_metrics']['building3d_aco'] for r in test_results) / len(test_results)
    avg_building3d_cp = sum(r['building3d_metrics']['building3d_cp'] for r in test_results) / len(test_results)
    avg_building3d_cr = sum(r['building3d_metrics']['building3d_cr'] for r in test_results) / len(test_results)
    avg_building3d_c_f1 = sum(r['building3d_metrics']['building3d_c_f1'] for r in test_results) / len(test_results)
    avg_building3d_ep = sum(r['building3d_metrics']['building3d_ep'] for r in test_results) / len(test_results)
    avg_building3d_er = sum(r['building3d_metrics']['building3d_er'] for r in test_results) / len(test_results)
    avg_building3d_e_f1 = sum(r['building3d_metrics']['building3d_e_f1'] for r in test_results) / len(test_results)
    
    # Print comprehensive results
    print(f"\nComprehensive Test Results:")
    print(f"Average Vertex RMSE: {avg_vertex_rmse:.6f}")
    print(f"Average Edge Precision: {avg_edge_precision:.6f}")
    print(f"Average Edge Recall: {avg_edge_recall:.6f}")
    print(f"Average Edge F1-Score: {avg_edge_f1:.6f}")
    print(f"\nBuilding3D Metrics:")
    print(f"ACO (Average Corner Offset): {avg_building3d_aco:.6f}")
    print(f"CP (Corner Precision): {avg_building3d_cp:.6f}")
    print(f"CR (Corner Recall): {avg_building3d_cr:.6f}")
    print(f"C-F1 (Corner F1): {avg_building3d_c_f1:.6f}")
    print(f"EP (Edge Precision): {avg_building3d_ep:.6f}")
    print(f"ER (Edge Recall): {avg_building3d_er:.6f}")
    print(f"E-F1 (Edge F1): {avg_building3d_e_f1:.6f}")

    # Log comprehensive metrics to W&B
    run.log({
        "eval_avg_vertex_rmse": avg_vertex_rmse,
        "eval_avg_edge_precision": avg_edge_precision,
        "eval_avg_edge_recall": avg_edge_recall,
        "eval_avg_edge_f1": avg_edge_f1,
        "eval_building3d_aco": avg_building3d_aco,
        "eval_building3d_cp": avg_building3d_cp,
        "eval_building3d_cr": avg_building3d_cr,
        "eval_building3d_c_f1": avg_building3d_c_f1,
        "eval_building3d_ep": avg_building3d_ep,
        "eval_building3d_er": avg_building3d_er,
        "eval_building3d_e_f1": avg_building3d_e_f1,
    })
    
    # Log a summary metric for easy comparison
    run.log({
        "Building3D_Overall_Score": (avg_building3d_c_f1 + avg_building3d_e_f1) / 2
    })
    
    # Print results
    
    print("\nTest Results:")
    print("-" * 40)
    for result in test_results:
        print(f"Sample {result['sample_index']+1}:")
        print(f"  Vertex RMSE: {result['vertex_rmse']:.6f}")
        print(f"  Edge Accuracy: {result['edge_accuracy']:.6f}")
        print(f"  Edge Precision: {result['edge_precision']:.6f}")
        print(f"  Edge Recall: {result['edge_recall']:.6f}")
        print(f"  Edge F1-Score: {result['edge_f1_score']:.6f}")
        print()
    
    print("\nModel saved as 'trained_model.pth'")
    
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

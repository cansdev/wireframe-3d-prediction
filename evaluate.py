import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib.pyplot as plt
import numpy as np
from demo_dataset.PCtoWFdataset import PCtoWFdataset
from models.PointCloudToWireframe import PointCloudToWireframe
from train import train_overfit_model, evaluate_model
from visualize.visualize_wireframe import visualize_prediction_comparison, visualize_edge_probabilities

def load_trained_model():
    """Load the pre-trained model from trained_model.pth"""
    print("Loading pre-trained model...")
    
    # Load data to get model parameters
    dataset = PCtoWFdataset(
        train_pc_dir='demo_dataset/train_dataset/point_cloud',
        train_wf_dir='demo_dataset/train_dataset/wireframe',
        test_pc_dir='demo_dataset/test_dataset/point_cloud',
        test_wf_dir='demo_dataset/test_dataset/wireframe'
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load training dataset to get model dimensions
    print("Loading training dataset to get model dimensions...")
    train_dataset = dataset.load_training_dataset()
    train_dataset.load_all_data()
    
    # Get max vertices from training data
    max_vertices = train_dataset.max_vertices
    
    # Create model with same architecture as training
    model = PointCloudToWireframe(input_dim=8, max_vertices=max_vertices).to(device)
    
    # Load trained weights
    if os.path.exists('trained_model.pth'):
        model.load_state_dict(torch.load('trained_model.pth', map_location=device))
        model.eval()
        print(f"✓ Pre-trained model loaded successfully from 'trained_model.pth'")
    else:
        raise FileNotFoundError("trained_model.pth not found! Please run main.py first to train the model.")
    
    return model, dataset, device, train_dataset

def create_individual_visualizations(model, dataset_obj, device, output_dir, dataset_name):
    """Create comprehensive visualizations for a single dataset"""
    print(f"Creating visualizations for {dataset_name}...")
    
    # Create dataset-specific output directory
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Create prediction comparison
    fig1 = visualize_prediction_comparison(dataset_obj, model, device)
    fig1.savefig(os.path.join(dataset_output_dir, f'{dataset_name}_prediction_comparison.png'), 
                 dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # Get predictions for edge probability visualization
    model.eval()
    with torch.no_grad():
        point_cloud_tensor = torch.FloatTensor(dataset_obj.normalized_point_cloud).unsqueeze(0).to(device)
        predictions = model(point_cloud_tensor)
        pred_edge_probs = predictions['edge_probs'].cpu().numpy()[0]
        edge_indices = predictions['edge_indices']
    
    # Create edge probability visualization
    fig2 = visualize_edge_probabilities(pred_edge_probs, edge_indices)
    fig2.savefig(os.path.join(dataset_output_dir, f'{dataset_name}_edge_probabilities.png'), 
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"✓ Saved visualizations for {dataset_name} in {dataset_output_dir}")

def analyze_individual_predictions(model, dataset_obj, device, dataset_name):
    """Analyze predictions for a single dataset"""
    model.eval()
    
    with torch.no_grad():
        # Get predictions
        point_cloud_tensor = torch.FloatTensor(dataset_obj.normalized_point_cloud).unsqueeze(0).to(device)
        predictions = model(point_cloud_tensor)
        
        # Extract predictions
        pred_vertices = predictions['vertices'].cpu().numpy()[0]
        pred_edge_probs = predictions['edge_probs'].cpu().numpy()[0]
        edge_indices = predictions['edge_indices']
        
        # Get actual number of vertices for this dataset
        actual_num_vertices = len(dataset_obj.vertices)
        pred_vertices_actual = pred_vertices[:actual_num_vertices]
        
        # Convert back to original scale
        pred_vertices_original = dataset_obj.spatial_scaler.inverse_transform(pred_vertices_actual)
        true_vertices = dataset_obj.vertices
        
        print(f"\n{dataset_name} Analysis:")
        print("-" * 40)
        
        # Vertex analysis
        vertex_errors = np.linalg.norm(pred_vertices_original - true_vertices, axis=1)
        print(f"Vertex Prediction Analysis:")
        print(f"  Average error: {vertex_errors.mean():.6f}")
        print(f"  Max error: {vertex_errors.max():.6f}")
        print(f"  Min error: {vertex_errors.min():.6f}")
        print(f"  Std deviation: {vertex_errors.std():.6f}")
        
        # Edge analysis  
        true_edges = set()
        for edge in dataset_obj.edges:
            true_edges.add((min(edge), max(edge)))
        
        predicted_edges = set()
        for idx, (i, j) in enumerate(edge_indices):
            if i < actual_num_vertices and j < actual_num_vertices and pred_edge_probs[idx] > 0.5:
                predicted_edges.add((min(i, j), max(i, j)))
        
        print(f"Edge Prediction Analysis:")
        print(f"  True edges: {len(true_edges)}")
        print(f"  Predicted edges: {len(predicted_edges)}")
        print(f"  Correct predictions: {len(true_edges & predicted_edges)}")
        print(f"  False positives: {len(predicted_edges - true_edges)}")
        print(f"  False negatives: {len(true_edges - predicted_edges)}")
        
        # Calculate metrics
        tp = len(true_edges & predicted_edges)
        fp = len(predicted_edges - true_edges)
        fn = len(true_edges - predicted_edges)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  Precision: {precision:.6f}")
        print(f"  Recall: {recall:.6f}")
        print(f"  F1-Score: {f1_score:.6f}")
        
        return {
            'vertex_rmse': np.sqrt(np.mean(vertex_errors**2)),
            'edge_precision': precision,
            'edge_recall': recall,
            'edge_f1_score': f1_score,
            'vertex_errors': vertex_errors,
            'pred_vertices': pred_vertices_original,
            'true_vertices': true_vertices,
            'edge_probs': pred_edge_probs,
            'edge_indices': edge_indices,
            'true_edges': true_edges,
            'predicted_edges': predicted_edges
        }

def create_comprehensive_visualizations_for_all():
    """Create comprehensive visualizations for all test datasets"""
    print("="*60)
    print("COMPREHENSIVE EVALUATION FOR ALL TEST DATASETS")
    print("="*60)
    
    # Load pre-trained model
    model, dataset, device, train_dataset = load_trained_model()
    
    print("\nLoading test datasets...")
    test_dataset = dataset.load_testing_dataset()
    test_dataset.load_all_data()
    
    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each test dataset individually
    print(f"\nProcessing {len(test_dataset.datasets)} test datasets...")
    
    all_results = []
    
    for i, individual_dataset in enumerate(test_dataset.datasets):
        dataset_name = f"test_dataset_{i+1}"
        
        # Analyze predictions
        analysis = analyze_individual_predictions(model, individual_dataset, device, dataset_name)
        all_results.append({
            'name': dataset_name,
            'analysis': analysis
        })
        
        # Create visualizations
        create_individual_visualizations(model, individual_dataset, device, output_dir, dataset_name)
    
    # Create summary report
    create_summary_report(all_results, output_dir)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, 'trained_model.pth'))
    
    print("\n" + "="*60)
    print("✓ COMPREHENSIVE EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nGenerated files in '{output_dir}' directory:")
    for result in all_results:
        dataset_name = result['name']
        print(f"  • {dataset_name}/")
        print(f"    - {dataset_name}_prediction_comparison.png")
        print(f"    - {dataset_name}_edge_probabilities.png")
    print("  • summary_report.txt")
    print("  • trained_model.pth")
    
    return all_results

def create_summary_report(results, output_dir):
    """Create a summary report of all test results"""
    report_path = os.path.join(output_dir, 'summary_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("COMPREHENSIVE TEST RESULTS SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        # Overall statistics
        all_vertex_rmse = [r['analysis']['vertex_rmse'] for r in results]
        all_precision = [r['analysis']['edge_precision'] for r in results]
        all_recall = [r['analysis']['edge_recall'] for r in results]
        all_f1 = [r['analysis']['edge_f1_score'] for r in results]
        
        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average Vertex RMSE: {np.mean(all_vertex_rmse):.6f} ± {np.std(all_vertex_rmse):.6f}\n")
        f.write(f"Average Edge Precision: {np.mean(all_precision):.6f} ± {np.std(all_precision):.6f}\n")
        f.write(f"Average Edge Recall: {np.mean(all_recall):.6f} ± {np.std(all_recall):.6f}\n")
        f.write(f"Average Edge F1-Score: {np.mean(all_f1):.6f} ± {np.std(all_f1):.6f}\n\n")
        
        # Individual results
        f.write("INDIVIDUAL RESULTS:\n")
        f.write("-" * 30 + "\n")
        for result in results:
            name = result['name']
            analysis = result['analysis']
            f.write(f"{name}:\n")
            f.write(f"  Vertex RMSE: {analysis['vertex_rmse']:.6f}\n")
            f.write(f"  Edge Precision: {analysis['edge_precision']:.6f}\n")
            f.write(f"  Edge Recall: {analysis['edge_recall']:.6f}\n")
            f.write(f"  Edge F1-Score: {analysis['edge_f1_score']:.6f}\n")
            f.write(f"  True Edges: {len(analysis['true_edges'])}\n")
            f.write(f"  Predicted Edges: {len(analysis['predicted_edges'])}\n\n")
    
    print(f"✓ Summary report saved to {report_path}")

def create_comprehensive_visualizations():
    """Original function kept for compatibility"""
    return create_comprehensive_visualizations_for_all()

if __name__ == "__main__":
    print("="*60)
    print("COMPREHENSIVE EVALUATION FOR ALL TEST DATASETS")
    print("="*60)
    
    try:
        # Create comprehensive visualizations for all test datasets
        all_results = create_comprehensive_visualizations_for_all()
        
        print("\n" + "="*60)
        print("✓ EVALUATION COMPLETE!")
        print("="*60)
        print("\nCheck the 'output' directory for:")
        print("  • Individual prediction comparisons for each test dataset")
        print("  • Edge probability distributions for each test dataset") 
        print("  • Summary report with all metrics")
        print("  • Trained model weights")
        print("\nThe model was trained on all training data and evaluated on all test data!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc() 
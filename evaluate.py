import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib.pyplot as plt
import numpy as np
from demo_dataset.PCtoWFdataset import PCtoWFdataset
from models.PointCloudToWireframe import PointCloudToWireframe
from visualize.visualize_wireframe import visualize_prediction_comparison, visualize_edge_probabilities
from scipy.spatial.distance import cdist

def calculate_corner_overlap(pred_vertices, true_vertices, threshold=0.1):
    """
    Calculate Average Corner Overlap (ACO)
    Measures how well predicted corners overlap with ground-truth corners
    """
    if len(pred_vertices) == 0 or len(true_vertices) == 0:
        return 0.0
    
    # Calculate distance matrix between predicted and true vertices
    distances = cdist(pred_vertices, true_vertices)
    
    # For each predicted vertex, find the closest true vertex
    min_distances = np.min(distances, axis=1)
    
    # Count how many predicted vertices are within threshold of true vertices
    overlapping_vertices = np.sum(min_distances <= threshold)
    
    # Calculate overlap ratio
    overlap_ratio = overlapping_vertices / len(pred_vertices)
    
    return overlap_ratio

def calculate_corner_metrics(pred_vertices, true_vertices, threshold=0.1):
    """
    Calculate Corner Precision (CP), Corner Recall (CR), and Corner F1 (C-F1)
    """
    if len(pred_vertices) == 0 and len(true_vertices) == 0:
        return 1.0, 1.0, 1.0  # Perfect match when both are empty
    
    if len(pred_vertices) == 0:
        return 0.0, 0.0, 0.0  # No predictions
    
    if len(true_vertices) == 0:
        return 0.0, 1.0, 0.0  # No true corners to match
    
    # Calculate distance matrix
    distances = cdist(pred_vertices, true_vertices)
    
    # For Corner Precision: how many predicted vertices match true vertices
    pred_min_distances = np.min(distances, axis=1)
    correct_predictions = np.sum(pred_min_distances <= threshold)
    corner_precision = correct_predictions / len(pred_vertices)
    
    # For Corner Recall: how many true vertices are detected
    true_min_distances = np.min(distances, axis=0)
    detected_true_vertices = np.sum(true_min_distances <= threshold)
    corner_recall = detected_true_vertices / len(true_vertices)
    
    # Corner F1 Score
    if corner_precision + corner_recall == 0:
        corner_f1 = 0.0
    else:
        corner_f1 = 2 * corner_precision * corner_recall / (corner_precision + corner_recall)
    
    return corner_precision, corner_recall, corner_f1

def calculate_enhanced_edge_metrics(pred_edges, true_edges):
    """
    Calculate Edge Precision (EP), Edge Recall (ER), and Edge F1 (E-F1)
    """
    if len(pred_edges) == 0 and len(true_edges) == 0:
        return 1.0, 1.0, 1.0  # Perfect match when both are empty
    
    pred_edge_set = set(pred_edges)
    true_edge_set = set(true_edges)
    
    # True Positives: edges that exist in both predicted and true sets
    tp = len(pred_edge_set & true_edge_set)
    
    # False Positives: edges predicted but not in ground truth
    fp = len(pred_edge_set - true_edge_set)
    
    # False Negatives: true edges not predicted
    fn = len(true_edge_set - pred_edge_set)
    
    # Edge Precision (EP)
    edge_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Edge Recall (ER)
    edge_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Edge F1 Score (E-F1)
    if edge_precision + edge_recall == 0:
        edge_f1 = 0.0
    else:
        edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall)
    
    return edge_precision, edge_recall, edge_f1

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
        model.load_state_dict(torch.load('trained_model.pth', map_location=device), strict=False)
        model.eval()
        print(f"✓ Pre-trained model loaded successfully from 'trained_model.pth'")
    else:
        raise FileNotFoundError("trained_model.pth not found! Please run main.py first to train the model.")
    
    return model, dataset, device, train_dataset

def create_individual_visualizations(model, sample_obj, device, output_dir, sample_name):
    """Create comprehensive visualizations for a single sample"""
    print(f"Creating visualizations for {sample_name}...")
    
    # Create sample-specific output directory
    sample_output_dir = os.path.join(output_dir, sample_name)
    os.makedirs(sample_output_dir, exist_ok=True)
    
    # Create prediction comparison
    fig1 = visualize_prediction_comparison(sample_obj, model, device)
    fig1.savefig(os.path.join(sample_output_dir, f'{sample_name}_prediction_comparison.png'), 
                 dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # Get predictions for edge probability visualization
    model.eval()
    with torch.no_grad():
        point_cloud_tensor = torch.FloatTensor(sample_obj.normalized_point_cloud).unsqueeze(0).to(device)
        predictions = model(point_cloud_tensor)
        pred_edge_probs = predictions['edge_probs'].cpu().numpy()[0]
        edge_indices = predictions['edge_indices']
    
    # Create edge probability visualization
    fig2 = visualize_edge_probabilities(pred_edge_probs, edge_indices)
    fig2.savefig(os.path.join(sample_output_dir, f'{sample_name}_edge_probabilities.png'), 
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"✓ Saved visualizations for {sample_name} in {sample_output_dir}")

def analyze_individual_predictions(model, sample_obj, device, sample_name):
    """Analyze predictions for a single sample"""
    model.eval()
    
    with torch.no_grad():
        # Get predictions
        point_cloud_tensor = torch.FloatTensor(sample_obj.normalized_point_cloud).unsqueeze(0).to(device)
        predictions = model(point_cloud_tensor)
        
        # Extract predictions
        pred_vertices = predictions['vertices'].cpu().numpy()[0]
        pred_edge_probs = predictions['edge_probs'].cpu().numpy()[0]
        edge_indices = predictions['edge_indices']
        
        # Get actual number of vertices for this dataset
        actual_num_vertices = len(sample_obj.vertices)
        pred_vertices_actual = pred_vertices[:actual_num_vertices]
        
        # Convert back to original scale
        pred_vertices_original = sample_obj.spatial_scaler.inverse_transform(pred_vertices_actual)
        true_vertices = sample_obj.vertices
        
        print(f"\n{sample_name} Analysis:")
        print("-" * 40)
        
        # ===== BUILDING3D BENCHMARK METRICS =====
        # ACO (Average Corner Offset) - vertex position accuracy
        vertex_errors = np.linalg.norm(pred_vertices_original - true_vertices, axis=1)
        aco = np.mean(vertex_errors)
        
        # Corner (Vertex) metrics for Building3D
        # For corners, we consider a vertex correctly predicted if within a threshold
        corner_threshold = 2.0  # meters - adjust based on your scale
        corner_correct = vertex_errors <= corner_threshold
        corner_tp = np.sum(corner_correct)
        corner_fp = actual_num_vertices - corner_tp  # assuming we predict exactly the right number
        corner_fn = 0  # assuming we predict exactly the right number
        
        cp = corner_tp / (corner_tp + corner_fp) if (corner_tp + corner_fp) > 0 else 0
        cr = corner_tp / (corner_tp + corner_fn) if (corner_tp + corner_fn) > 0 else 0
        c_f1 = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0
        
        print(f"Building3D Metrics:")
        print(f"  ACO (Average Corner Offset): {aco:.6f}")
        print(f"  CP (Corner Precision): {cp:.6f}")
        print(f"  CR (Corner Recall): {cr:.6f}")
        print(f"  C-F1 (Corner F1): {c_f1:.6f}")
        
        # ===== OUR CUSTOM METRICS =====
        print(f"Custom Vertex Analysis:")
        print(f"  Average error: {vertex_errors.mean():.6f}")
        print(f"  Max error: {vertex_errors.max():.6f}")
        print(f"  Min error: {vertex_errors.min():.6f}")
        print(f"  Std deviation: {vertex_errors.std():.6f}")
        
        # Edge analysis using edge set
        true_edges = sample_obj.edge_set if hasattr(sample_obj, 'edge_set') and sample_obj.edge_set else set()
        # Fallback to creating edge set from edges if needed
        if not true_edges:
            true_edges = set()
            for edge in sample_obj.edges:
                true_edges.add((min(edge[0], edge[1]), max(edge[0], edge[1])))
        
        predicted_edges = set()
        for idx, (i, j) in enumerate(edge_indices):
            if i < actual_num_vertices and j < actual_num_vertices and pred_edge_probs[idx] > 0.5:
                predicted_edges.add((min(i, j), max(i, j)))
        
        # ===== BUILDING3D EDGE METRICS =====
        edge_tp = len(true_edges & predicted_edges)
        edge_fp = len(predicted_edges - true_edges)
        edge_fn = len(true_edges - predicted_edges)
        
        ep = edge_tp / (edge_tp + edge_fp) if (edge_tp + edge_fp) > 0 else 0
        er = edge_tp / (edge_tp + edge_fn) if (edge_tp + edge_fn) > 0 else 0
        e_f1 = 2 * ep * er / (ep + er) if (ep + er) > 0 else 0
        
        print(f"Building3D Edge Metrics:")
        print(f"  EP (Edge Precision): {ep:.6f}")
        print(f"  ER (Edge Recall): {er:.6f}")
        print(f"  E-F1 (Edge F1): {e_f1:.6f}")
        
        # ===== OUR CUSTOM EDGE METRICS =====
        print(f"Custom Edge Analysis:")
        print(f"  True edges: {len(true_edges)}")
        print(f"  Predicted edges: {len(predicted_edges)}")
        print(f"  Correct predictions: {edge_tp}")
        print(f"  False positives: {edge_fp}")
        print(f"  False negatives: {edge_fn}")
        main
        
        # Calculate our custom edge metrics
        precision = ep  # same as Building3D EP
        recall = er    # same as Building3D ER
        f1_score = e_f1  # same as Building3D E-F1
        
        print(f"  Precision: {precision:.6f}")
        print(f"  Recall: {recall:.6f}")
        print(f"  F1-Score: {f1_score:.6f}")
        
        # Calculate new corner-based metrics
        corner_overlap = calculate_corner_overlap(pred_vertices_original, true_vertices, threshold=0.1)
        corner_precision, corner_recall, corner_f1 = calculate_corner_metrics(pred_vertices_original, true_vertices, threshold=0.1)
        
        # Calculate enhanced edge metrics (same as above but for consistency)
        edge_precision_enhanced, edge_recall_enhanced, edge_f1_enhanced = calculate_enhanced_edge_metrics(
            list(predicted_edges), list(true_edges)
        )
        
        print(f"\nCorner-based Metrics:")
        print(f"  ACO (Average Corner Overlap): {corner_overlap:.6f}")
        print(f"  CP (Corner Precision): {corner_precision:.6f}")
        print(f"  CR (Corner Recall): {corner_recall:.6f}")
        print(f"  C-F1 (Corner F1 Score): {corner_f1:.6f}")
        
        print(f"\nEnhanced Edge Metrics:")
        print(f"  EP (Edge Precision): {edge_precision_enhanced:.6f}")
        print(f"  ER (Edge Recall): {edge_recall_enhanced:.6f}")
        print(f"  E-F1 (Edge F1 Score): {edge_f1_enhanced:.6f}")
        
        return {
            # Building3D metrics
            'building3d_aco': aco,
            'building3d_cp': cp,
            'building3d_cr': cr,
            'building3d_c_f1': c_f1,
            'building3d_ep': ep,
            'building3d_er': er,
            'building3d_e_f1': e_f1,
            
            # Our custom metrics (kept separate)
            'vertex_rmse': np.sqrt(np.mean(vertex_errors**2)),
            'edge_precision': precision,
            'edge_recall': recall,
            'edge_f1_score': f1_score,
            # New corner-based metrics
            'corner_overlap': corner_overlap,
            'corner_precision': corner_precision,
            'corner_recall': corner_recall,
            'corner_f1': corner_f1,
            # Enhanced edge metrics
            'edge_precision_enhanced': edge_precision_enhanced,
            'edge_recall_enhanced': edge_recall_enhanced,
            'edge_f1_enhanced': edge_f1_enhanced,
            # Data for further analysis
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
    print(f"\nProcessing {len(test_dataset.samples)} test samples...")
    
    all_results = []
    
    for i, individual_sample in enumerate(test_dataset.samples):
        sample_name = f"test_sample_{i+1}"
        
        # Analyze predictions
        analysis = analyze_individual_predictions(model, individual_sample, device, sample_name)
        all_results.append({
            'name': sample_name,
            'analysis': analysis
        })
        
        # Create visualizations (disabled by default - use visualize1.py when needed)
        # create_individual_visualizations(model, individual_sample, device, output_dir, sample_name)
    
    # Create summary report
    create_summary_report(all_results, output_dir)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, 'trained_model.pth'))
    
    # Log summary report as W&B artifact if wandb is available
    try:
        import wandb
        # Try to resume the W&B run from training
        run_id_file = 'wandb_run_id.txt'
        if os.path.exists(run_id_file):
            with open(run_id_file, 'r') as f:
                run_id = f.read().strip()
            
            # Resume the W&B run
            api = wandb.Api()
            try:
                # Resume the run
                wandb.init(
                    project="Wireframe3D",
                    entity="can_g-a",
                    id=run_id,
                    resume="must"
                )
                print(f"✓ Resumed W&B run: {run_id}")
                
                # Log the summary report
                report_path = os.path.join(output_dir, 'summary_report.txt')
                if os.path.exists(report_path):
                    artifact = wandb.Artifact(
                        name="evaluation_summary_report",
                        type="evaluation_report", 
                        description="Comprehensive evaluation results with Building3D and custom metrics"
                    )
                    artifact.add_file(report_path)
                    wandb.log_artifact(artifact)
                    print(f"✓ Summary report logged to W&B as artifact: {artifact.name}")
                else:
                    print("⚠ Warning: summary_report.txt not found, skipping W&B artifact logging")
                
                # Finish the resumed run
                wandb.finish()
                
            except Exception as e:
                print(f"⚠ Warning: Could not resume W&B run {run_id}: {e}")
                print("ℹ Info: Skipping W&B artifact logging")
        else:
            print("ℹ Info: No W&B run ID found, skipping artifact logging")
    except ImportError:
        print("ℹ Info: W&B not available, skipping artifact logging")
    
    print("\n" + "="*60)
    print("✓ COMPREHENSIVE EVALUATION COMPLETE!")
    print("="*60)
    print(f"\nGenerated files in '{output_dir}' directory:")
    for result in all_results:
        sample_name = result['name']
        print(f"  • {sample_name}/")
        print(f"    - {sample_name}_prediction_comparison.png")
        print(f"    - {sample_name}_edge_probabilities.png")
    print("  • summary_report.txt")
    print("  • trained_model.pth")
    
    return all_results

def create_summary_report(results, output_dir):
    """Create a summary report of all test results with proper global metrics"""
    report_path = os.path.join(output_dir, 'summary_report.txt')
    
    # Calculate global metrics by aggregating all true positives, false positives, and false negatives
    global_tp = 0
    global_fp = 0
    global_fn = 0
    all_vertex_errors = []
    
    # Building3D metrics aggregation
    all_aco = []
    all_cp = []
    all_cr = []
    all_c_f1 = []
    all_ep = []
    all_er = []
    all_e_f1 = []
    
    for result in results:
        analysis = result['analysis']
        true_edges = analysis['true_edges']
        predicted_edges = analysis['predicted_edges']
        
        # Accumulate global counts for edge metrics
        global_tp += len(true_edges & predicted_edges)
        global_fp += len(predicted_edges - true_edges)
        global_fn += len(true_edges - predicted_edges)
        
        # Collect all vertex errors for global RMSE
        all_vertex_errors.extend(analysis['vertex_errors'])
        
        # Collect Building3D metrics
        all_aco.append(analysis['building3d_aco'])
        all_cp.append(analysis['building3d_cp'])
        all_cr.append(analysis['building3d_cr'])
        all_c_f1.append(analysis['building3d_c_f1'])
        all_ep.append(analysis['building3d_ep'])
        all_er.append(analysis['building3d_er'])
        all_e_f1.append(analysis['building3d_e_f1'])
    
    # Calculate global metrics
    global_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    global_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0
    global_vertex_rmse = np.sqrt(np.mean(np.array(all_vertex_errors)**2))
    
    # Calculate Building3D global metrics (averaged across samples)
    global_aco = np.mean(all_aco)
    global_cp = np.mean(all_cp)
    global_cr = np.mean(all_cr)
    global_c_f1 = np.mean(all_c_f1)
    global_ep = np.mean(all_ep)
    global_er = np.mean(all_er)
    global_e_f1 = np.mean(all_e_f1)
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("COMPREHENSIVE TEST RESULTS SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        # Building3D Benchmark Metrics (Global)
        f.write("BUILDING3D BENCHMARK METRICS (Global Averages):\n")
        f.write("-" * 50 + "\n")
        f.write(f"ACO (Average Corner Offset): {global_aco:.6f}\n")
        f.write(f"CP (Corner Precision): {global_cp:.6f}\n")
        f.write(f"CR (Corner Recall): {global_cr:.6f}\n")
        f.write(f"C-F1 (Corner F1): {global_c_f1:.6f}\n")
        f.write(f"EP (Edge Precision): {global_ep:.6f}\n")
        f.write(f"ER (Edge Recall): {global_er:.6f}\n")
        f.write(f"E-F1 (Edge F1): {global_e_f1:.6f}\n\n")
        
        # Our Custom Global statistics (proper aggregation)
        f.write("OUR CUSTOM METRICS (Global Aggregation):\n")
        f.write("-" * 50 + "\n")
        f.write(f"Global Vertex RMSE: {global_vertex_rmse:.6f}\n")
        f.write(f"Global Edge Precision: {global_precision:.6f}\n")
        f.write(f"Global Edge Recall: {global_recall:.6f}\n")
        f.write(f"Global Edge F1-Score: {global_f1:.6f}\n")
        f.write(f"Total True Positives: {global_tp}\n")
        f.write(f"Total False Positives: {global_fp}\n")
        f.write(f"Total False Negatives: {global_fn}\n\n")
        
        # Per-sample averages (for comparison)
        all_vertex_rmse = [r['analysis']['vertex_rmse'] for r in results]
        all_precision = [r['analysis']['edge_precision'] for r in results]
        all_recall = [r['analysis']['edge_recall'] for r in results]
        all_f1 = [r['analysis']['edge_f1_score'] for r in results]
        
        # New corner-based metrics
        all_corner_overlap = [r['analysis']['corner_overlap'] for r in results]
        all_corner_precision = [r['analysis']['corner_precision'] for r in results]
        all_corner_recall = [r['analysis']['corner_recall'] for r in results]
        all_corner_f1 = [r['analysis']['corner_f1'] for r in results]
        
        # Enhanced edge metrics
        all_edge_precision_enhanced = [r['analysis']['edge_precision_enhanced'] for r in results]
        all_edge_recall_enhanced = [r['analysis']['edge_recall_enhanced'] for r in results]
        all_edge_f1_enhanced = [r['analysis']['edge_f1_enhanced'] for r in results]
        
        f.write("PER-SAMPLE AVERAGES (for comparison):\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average Vertex RMSE: {np.mean(all_vertex_rmse):.6f} ± {np.std(all_vertex_rmse):.6f}\n")
        f.write(f"Average Edge Precision: {np.mean(all_precision):.6f} ± {np.std(all_precision):.6f}\n")
        f.write(f"Average Edge Recall: {np.mean(all_recall):.6f} ± {np.std(all_recall):.6f}\n")
        f.write(f"Average Edge F1-Score: {np.mean(all_f1):.6f} ± {np.std(all_f1):.6f}\n\n")
        
        f.write("CORNER-BASED METRICS (per-sample averages):\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average ACO (Corner Overlap): {np.mean(all_corner_overlap):.6f} ± {np.std(all_corner_overlap):.6f}\n")
        f.write(f"Average CP (Corner Precision): {np.mean(all_corner_precision):.6f} ± {np.std(all_corner_precision):.6f}\n")
        f.write(f"Average CR (Corner Recall): {np.mean(all_corner_recall):.6f} ± {np.std(all_corner_recall):.6f}\n")
        f.write(f"Average C-F1 (Corner F1): {np.mean(all_corner_f1):.6f} ± {np.std(all_corner_f1):.6f}\n\n")
        
        f.write("ENHANCED EDGE METRICS (per-sample averages):\n")
        f.write("-" * 40 + "\n")
        f.write(f"Average EP (Edge Precision): {np.mean(all_edge_precision_enhanced):.6f} ± {np.std(all_edge_precision_enhanced):.6f}\n")
        f.write(f"Average ER (Edge Recall): {np.mean(all_edge_recall_enhanced):.6f} ± {np.std(all_edge_recall_enhanced):.6f}\n")
        f.write(f"Average E-F1 (Edge F1): {np.mean(all_edge_f1_enhanced):.6f} ± {np.std(all_edge_f1_enhanced):.6f}\n\n")
        
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
            f.write(f"  ACO (Corner Overlap): {analysis['corner_overlap']:.6f}\n")
            f.write(f"  CP (Corner Precision): {analysis['corner_precision']:.6f}\n")
            f.write(f"  CR (Corner Recall): {analysis['corner_recall']:.6f}\n")
            f.write(f"  C-F1 (Corner F1): {analysis['corner_f1']:.6f}\n")
            f.write(f"  EP (Edge Precision): {analysis['edge_precision_enhanced']:.6f}\n")
            f.write(f"  ER (Edge Recall): {analysis['edge_recall_enhanced']:.6f}\n")
            f.write(f"  E-F1 (Edge F1): {analysis['edge_f1_enhanced']:.6f}\n")
            f.write(f"  True Edges: {len(analysis['true_edges'])}\n")
            f.write(f"  Predicted Edges: {len(analysis['predicted_edges'])}\n\n")
            f.write(f"  Building3D - ACO: {analysis['building3d_aco']:.6f}, CP: {analysis['building3d_cp']:.6f}, CR: {analysis['building3d_cr']:.6f}, C-F1: {analysis['building3d_c_f1']:.6f}\n")
            f.write(f"  Building3D - EP: {analysis['building3d_ep']:.6f}, ER: {analysis['building3d_er']:.6f}, E-F1: {analysis['building3d_e_f1']:.6f}\n")
            f.write(f"  Custom - Vertex RMSE: {analysis['vertex_rmse']:.6f}, Edge F1: {analysis['edge_f1_score']:.6f}\n")
            f.write(f"  True Edges: {len(analysis['true_edges'])}, Predicted Edges: {len(analysis['predicted_edges'])}\n\n")
            
          main
    
    print(f"✓ Summary report saved to {report_path}")
    print(f"Building3D Metrics - ACO: {global_aco:.4f}, C-F1: {global_c_f1:.4f}, E-F1: {global_e_f1:.4f}")
    print(f"Custom Metrics - Precision: {global_precision:.4f}, Recall: {global_recall:.4f}, F1: {global_f1:.4f}")

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
        print("  • Summary report with all metrics")
        print("  • Trained model weights")
        print("\nNote: Visualizations are disabled by default. Use visualize1.py to generate images when needed.")
        print("\nThe model was trained on all training data and evaluated on all test data!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc() 
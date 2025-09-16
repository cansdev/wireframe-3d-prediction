import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from eval.ap_calculator import APCalculator
from datasets import building3d, build_dataset
from models.PointCloudToWireframe import PointCloudToWireframe


def cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    cfg = EasyDict(new_config)
    return cfg

def visualize_point_cloud(point_cloud, title="Point Cloud", max_points=1000):
    """Visualize point cloud data"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample points if too many
    if len(point_cloud) > max_points:
        indices = np.random.choice(len(point_cloud), max_points, replace=False)
        sampled_points = point_cloud[indices]
    else:
        sampled_points = point_cloud
    
    # Extract coordinates and colors
    x, y, z = sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2]
    colors = sampled_points[:, 3:7] / 255.0  # RGB values
    
    # Plot points
    ax.scatter(x, y, z, c=colors[:, :3], s=1, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    return fig, ax

def visualize_wireframe(vertices, edges, title="Wireframe", color='blue'):
    """Visualize wireframe structure"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='red', s=50, alpha=0.8, label='Vertices')
    
    # Plot edges
    for edge in edges:
        v1, v2 = edge[0], edge[1]
        x_vals = [vertices[v1, 0], vertices[v2, 0]]
        y_vals = [vertices[v1, 1], vertices[v2, 1]]
        z_vals = [vertices[v1, 2], vertices[v2, 2]]
        ax.plot(x_vals, y_vals, z_vals, color=color, linewidth=2, alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    return fig, ax

def visualize_prediction_comparison(sample_data, model, device):
    """Compare predicted vs actual wireframes using APCalculator blueprint"""
    model.eval()
    
    with torch.no_grad():
        # Get predictions using new dataset format
        point_cloud_tensor = torch.FloatTensor(sample_data['point_clouds']).unsqueeze(0).to(device)
        vertex_count = torch.tensor([len(sample_data['wf_vertices'])], dtype=torch.long).to(device)
        predictions = model(point_cloud_tensor, vertex_count)
        
        # Extract predictions
        pred_vertices = predictions['vertices'].cpu().numpy()[0]
        pred_edge_probs = predictions['edge_probs'].cpu().numpy()[0]
        edge_indices = predictions['edge_indices'][0]
        
        # Filter edges by probability threshold (APCalculator blueprint)
        mask = pred_edge_probs > 0.5
        pd_edges = np.array(edge_indices)[mask]
        
        # Get ground truth
        gt_vertices = sample_data['wf_vertices']
        gt_edges = sample_data['wf_edges'].astype(np.int64)
        
        # Build batch exactly as APCalculator blueprint
        if len(pd_edges) > 0:
            pd_edges_vertices = np.stack((pred_vertices[pd_edges[:, 0]], pred_vertices[pd_edges[:, 1]]), axis=1)
            pd_edges_vertices = pd_edges_vertices[np.arange(pd_edges_vertices.shape[0])[:, np.newaxis], np.flip(np.argsort(pd_edges_vertices[:, :, -1]), axis=1)]
        else:
            pd_edges_vertices = np.empty((0, 2, 3))
        
        if len(gt_edges) > 0:
            gt_edges_vertices = np.stack((gt_vertices[gt_edges[:, 0]], gt_vertices[gt_edges[:, 1]]), axis=1)
            gt_edges_vertices = gt_edges_vertices[np.arange(gt_edges_vertices.shape[0])[:, np.newaxis], np.flip(np.argsort(gt_edges_vertices[:, :, -1]), axis=1)]
        else:
            gt_edges_vertices = np.empty((0, 2, 3))
        
        # Use APCalculator for evaluation
        ap_calculator = APCalculator(distance_thresh=1)
        
        batch = dict()
        batch['predicted_vertices'] = pred_vertices[np.newaxis, :]
        batch['predicted_edges'] = pd_edges[np.newaxis, :]
        batch['pred_edges_vertices'] = pd_edges_vertices.reshape((1, -1, 2, 3))
        
        batch['wf_vertices'] = gt_vertices[np.newaxis, :]
        batch['wf_edges'] = gt_edges[np.newaxis, :]
        batch['wf_edges_vertices'] = gt_edges_vertices.reshape((1, -1, 2, 3))
        
        ap_calculator.compute_metrics(batch)
        ap_calculator.output_accuracy()
    
    # Create comparison visualization
    fig = plt.figure(figsize=(20, 8))
    
    # Original wireframe
    ax1 = fig.add_subplot(131, projection='3d')
    vertices = gt_vertices
    edges = gt_edges
    
    ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='red', s=50, alpha=0.8, label='Vertices')
    
    for edge in edges:
        v1, v2 = edge[0], edge[1]
        x_vals = [vertices[v1, 0], vertices[v2, 0]]
        y_vals = [vertices[v1, 1], vertices[v2, 1]]
        z_vals = [vertices[v1, 2], vertices[v2, 2]]
        ax1.plot(x_vals, y_vals, z_vals, color='blue', linewidth=2, alpha=0.7)
    
    ax1.set_title('Ground Truth Wireframe')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # Predicted wireframe
    ax2 = fig.add_subplot(132, projection='3d')
    
    ax2.scatter(pred_vertices[:, 0], pred_vertices[:, 1], pred_vertices[:, 2], 
               c='red', s=50, alpha=0.8, label='Vertices')
    
    for edge in pd_edges:
        v1, v2 = edge[0], edge[1]
        x_vals = [pred_vertices[v1, 0], pred_vertices[v2, 0]]
        y_vals = [pred_vertices[v1, 1], pred_vertices[v2, 1]]
        z_vals = [pred_vertices[v1, 2], pred_vertices[v2, 2]]
        ax2.plot(x_vals, y_vals, z_vals, color='green', linewidth=2, alpha=0.7)
    
    ax2.set_title('Predicted Wireframe')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    # Overlay comparison
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Plot both sets of vertices
    ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='red', s=50, alpha=0.8, label='True Vertices')
    ax3.scatter(pred_vertices[:, 0], pred_vertices[:, 1], pred_vertices[:, 2], 
               c='orange', s=30, alpha=0.6, label='Pred Vertices')
    
    # Plot true edges in blue
    for edge in edges:
        v1, v2 = edge[0], edge[1]
        x_vals = [vertices[v1, 0], vertices[v2, 0]]
        y_vals = [vertices[v1, 1], vertices[v2, 1]]
        z_vals = [vertices[v1, 2], vertices[v2, 2]]
        ax3.plot(x_vals, y_vals, z_vals, color='blue', linewidth=3, alpha=0.7, label='True Edges' if edge is edges[0] else "")
    
    # Plot predicted edges in green
    for i, edge in enumerate(pd_edges):
        v1, v2 = edge[0], edge[1]
        x_vals = [pred_vertices[v1, 0], pred_vertices[v2, 0]]
        y_vals = [pred_vertices[v1, 1], pred_vertices[v2, 1]]
        z_vals = [pred_vertices[v1, 2], pred_vertices[v2, 2]]
        ax3.plot(x_vals, y_vals, z_vals, color='green', linewidth=2, alpha=0.7, 
                linestyle='--', label='Pred Edges' if i == 0 else "")
    
    ax3.set_title('Comparison Overlay')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    plt.tight_layout()
    return fig

def plot_training_loss(loss_history):
    """Plot training loss curve"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(loss_history, 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Time')
    ax.grid(True, alpha=0.3)
    
    # Add smoothed curve
    if len(loss_history) > 50:
        window_size = max(1, len(loss_history) // 50)
        smoothed = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
        ax.plot(range(window_size//2, len(loss_history) - window_size//2), smoothed, 'r-', 
                linewidth=3, alpha=0.7, label='Smoothed')
        ax.legend()
    
    return fig

def visualize_edge_probabilities(edge_probs, edge_indices, threshold=0.5):
    """Visualize edge probability distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram of edge probabilities
    ax1.hist(edge_probs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax1.set_xlabel('Edge Probability')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Edge Probabilities')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Edge probabilities sorted
    sorted_probs = np.sort(edge_probs)[::-1]
    # Use 1-based indexing for logarithmic scale (log(0) is undefined)
    x_indices = np.arange(1, len(sorted_probs) + 1)
    ax2.plot(x_indices, sorted_probs, 'b-', linewidth=2)
    ax2.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax2.set_xlabel('Edge Index (sorted by probability) - Log Scale')
    ax2.set_ylabel('Edge Probability')
    ax2.set_title('Sorted Edge Probabilities')
    ax2.set_xscale('log', base=2)  # Make x-axis logarithmic with base 100
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Load dataset using blueprint
    dataset_config = cfg_from_yaml_file('datasets/dataset_config.yaml')
    building3D_dataset = build_dataset(dataset_config.Building3D)
    
    # Get a sample for visualization
    sample = building3D_dataset['train'][0]
    
    # Visualize point cloud
    fig1 = visualize_point_cloud(sample['point_clouds'], "Original Point Cloud")
    plt.savefig('point_cloud_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Visualize original wireframe
    fig2 = visualize_wireframe(sample['wf_vertices'], sample['wf_edges'], "Original Wireframe")
    plt.savefig('wireframe_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    
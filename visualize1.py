import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from eval.ap_calculator import APCalculator
from datasets import building3d, build_dataset
from models.PointCloudToWireframe import PointCloudToWireframe
from visualize.visualize_wireframe import visualize_prediction_comparison, visualize_edge_probabilities
import matplotlib.pyplot as plt


def cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    cfg = EasyDict(new_config)
    return cfg


def load_trained_model_for_visualization():
    dataset_config = cfg_from_yaml_file('datasets/dataset_config.yaml')
    building3D_dataset = build_dataset(dataset_config.Building3D)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sample = building3D_dataset['train'][0]
    input_dim = sample['point_clouds'].shape[1]
    max_vertices = len(sample['wf_vertices'])
    
    model = PointCloudToWireframe(input_dim=input_dim, max_vertices=max_vertices).to(device)
    if os.path.exists('trained_model.pth'):
        model.load_state_dict(torch.load('trained_model.pth', map_location=device), strict=False)
        model.eval()
    else:
        raise FileNotFoundError('trained_model.pth not found')
    return model, building3D_dataset, device


def create_individual_visualizations(model, sample_data, device, output_dir, sample_name):
    """Create comprehensive visualizations for a single sample"""
    
    # Create sample-specific output directory
    sample_output_dir = os.path.join(output_dir, sample_name)
    os.makedirs(sample_output_dir, exist_ok=True)
    
    # Create prediction comparison
    fig1 = visualize_prediction_comparison(sample_data, model, device)
    fig1.savefig(os.path.join(sample_output_dir, f'{sample_name}_prediction_comparison.png'), 
                 dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # Get predictions for edge probability visualization
    model.eval()
    with torch.no_grad():
        point_cloud_tensor = torch.FloatTensor(sample_data['point_clouds']).unsqueeze(0).to(device)
        vertex_count = torch.tensor([len(sample_data['wf_vertices'])], dtype=torch.long).to(device)
        predictions = model(point_cloud_tensor, vertex_count)
        pred_edge_probs = predictions['edge_probs'].cpu().numpy()[0]
        edge_indices = predictions['edge_indices'][0]
    
    # Create edge probability visualization
    fig2 = visualize_edge_probabilities(pred_edge_probs, edge_indices)
    fig2.savefig(os.path.join(sample_output_dir, f'{sample_name}_edge_probabilities.png'), 
                 dpi=150, bbox_inches='tight')
    plt.close(fig2)


def evaluate_with_ap_calculator(model, selected_dataset, device):
    """Use APCalculator blueprint for evaluation"""
    ap_calculator = APCalculator(distance_thresh=1)
    
    with torch.no_grad():
        for i in range(len(selected_dataset)):
            sample_data = selected_dataset[i]
            
            # Model prediction
            point_cloud_tensor = torch.FloatTensor(sample_data['point_clouds']).unsqueeze(0).to(device)
            vertex_count = torch.tensor([len(sample_data['wf_vertices'])], dtype=torch.long).to(device)
            predictions = model(point_cloud_tensor, vertex_count)
            
            pred_vertices = predictions['vertices'].cpu().numpy()[0]
            edge_indices = predictions['edge_indices'][0]
            edge_probs = predictions['edge_probs'].cpu().numpy()[0]
            
            # Filter edges by probability threshold
            mask = edge_probs > 0.5
            pd_edges = np.array(edge_indices)[mask]
            
            # Get ground truth
            gt_vertices = sample_data['wf_vertices']
            gt_edges = sample_data['wf_edges'].astype(np.int64)
            
            # Build batch exactly as blueprint
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
            
            batch = dict()
            batch['predicted_vertices'] = pred_vertices[np.newaxis, :]
            batch['predicted_edges'] = pd_edges[np.newaxis, :]
            batch['pred_edges_vertices'] = pd_edges_vertices.reshape((1, -1, 2, 3))
            
            batch['wf_vertices'] = gt_vertices[np.newaxis, :]
            batch['wf_edges'] = gt_edges[np.newaxis, :]
            batch['wf_edges_vertices'] = gt_edges_vertices.reshape((1, -1, 2, 3))
            
            ap_calculator.compute_metrics(batch)
    
    ap_calculator.output_accuracy()


def main():
    model, building3D_dataset, device = load_trained_model_for_visualization()
    
    # Ask user which dataset to visualize
    dataset_choice = input("Enter your choice (1 or 2): ").strip()
    
    if dataset_choice == "1":
        # Use train dataset
        selected_dataset = building3D_dataset['train']
        dataset_name = "train"
        sample_prefix = "train_sample"
        
    else:
        # Default to test dataset
        selected_dataset = building3D_dataset['test']
        dataset_name = "test"
        sample_prefix = "test_sample"
    
    # Evaluate using APCalculator blueprint
    print(f"Evaluating {dataset_name} dataset with APCalculator...")
    evaluate_with_ap_calculator(model, selected_dataset, device)
    
    # Create visualizations for user-selected samples
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    user_input = input("Sample indices: ").strip()
    
    if user_input.lower() == 'all':
        sample_indices = list(range(len(selected_dataset)))
    else:
        try:
            sample_indices = [int(x.strip()) - 1 for x in user_input.split(',') if x.strip()]
            # Validate indices
            sample_indices = [i for i in sample_indices if 0 <= i < len(selected_dataset)]
        except ValueError:
            sample_indices = [0]
    
    if not sample_indices:    
        sample_indices = [0]
    
    for i in sample_indices:
        sample_data = selected_dataset[i]
        create_individual_visualizations(model, sample_data, device, output_dir, f'{sample_prefix}_{i+1}')

if __name__ == "__main__":
    main()

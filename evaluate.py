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


def cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    cfg = EasyDict(new_config)
    return cfg


def evaluate_with_ap_calculator():
    # Load dataset configuration (blueprint)
    dataset_config = cfg_from_yaml_file('datasets/dataset_config.yaml')
    
    # Build dataset with preprocessing (blueprint)
    building3D_dataset = build_dataset(dataset_config.Building3D)
    
    # Create test loader (blueprint)
    test_loader = DataLoader(
        building3D_dataset['test'], 
        batch_size=3, 
        shuffle=False, 
        drop_last=False, 
        collate_fn=building3D_dataset['test'].collate_batch
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load trained model
    sample = building3D_dataset['train'][0]
    input_dim = sample['point_clouds'].shape[1]
    
    # Extract max_vertices from the trained model's final layer
    if os.path.exists('trained_model.pth'):
        state_dict = torch.load('trained_model.pth', map_location=device)
        # Get the final layer weight shape: [max_vertices * vertex_dim, 1024]
        final_layer_weight_shape = state_dict['vertex_predictor.final_layer.weight'].shape
        max_vertices = final_layer_weight_shape[0] // 4  # vertex_dim = 4 (x, y, z, existence)
        
        model = PointCloudToWireframe(input_dim=input_dim, max_vertices=max_vertices).to(device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    else:
        raise FileNotFoundError('trained_model.pth not found')
    
    ap_calculator = APCalculator(distance_thresh=1)
    
    with torch.no_grad():
        for batch_data in test_loader:
            # Extract data from batch
            point_clouds = batch_data['point_clouds']
            wf_vertices = batch_data['wf_vertices']
            wf_edges = batch_data['wf_edges']
            
            # Model prediction
            vertex_counts = torch.tensor([len(vertices) for vertices in wf_vertices], dtype=torch.long).to(device)
            predictions = model(point_clouds.to(device), vertex_counts)
            
            # Process each sample in batch
            for i in range(len(wf_vertices)):
                # Get predictions for this sample
                pred_vertices = predictions['vertices'][i].cpu().numpy()
                edge_indices = predictions['edge_indices'][i]
                edge_probs = predictions['edge_probs'][i].cpu().numpy()
                
                # Filter edges by probability threshold
                mask = edge_probs > 0.5
                pd_edges = np.array(edge_indices)[mask]
                
                # Get ground truth
                gt_vertices = wf_vertices[i].numpy()
                gt_edges = wf_edges[i].numpy().astype(np.int64)
                
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


if __name__ == '__main__':
    evaluate_with_ap_calculator()



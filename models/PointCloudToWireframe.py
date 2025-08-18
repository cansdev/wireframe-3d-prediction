
import torch
import torch.nn as nn
from models.EdgePredictor import EdgePredictor
from models.PointNetEncoder import PointNetEncoder
from models.VertexPredictor import VertexPredictor


class PointCloudToWireframe(nn.Module):
    """Complete model for point cloud to wireframe prediction with dynamic vertex count"""
    
    def __init__(self, input_dim=8, max_vertices=64):
        super(PointCloudToWireframe, self).__init__()
        
        self.max_vertices = max_vertices
        
        # Point cloud encoder
        self.encoder = PointNetEncoder(input_dim=input_dim)
        
        # Dynamic vertex predictor
        self.vertex_predictor = VertexPredictor(
            global_feature_dim=512, 
            max_vertices=max_vertices
        )
        
        # Edge predictor
        self.edge_predictor = EdgePredictor(vertex_dim=3)
        
    def forward(self, point_cloud, target_vertex_counts=None):
        # Encode point cloud
        global_features, point_features = self.encoder(point_cloud)
        
        # Predict vertices with dynamic count
        vertex_output = self.vertex_predictor(global_features, target_vertex_counts)
        predicted_vertices = vertex_output['vertices']
        
        # During training, apply hard constraint on vertex count
        if self.training and target_vertex_counts is not None:
            # Mask out vertices beyond target count
            batch_size = predicted_vertices.shape[0]
            for i in range(batch_size):
                count = target_vertex_counts[i].item()
                if count < self.max_vertices:
                    # Zero out vertices beyond the target count
                    predicted_vertices[i, count:, :] = 0.0
        
        # For edge prediction, we need to mask out unused vertices
        batch_size = predicted_vertices.shape[0]
        
        # Get actual vertex counts for each sample in batch
        if self.training and target_vertex_counts is not None:
            actual_counts = target_vertex_counts
        else:
            actual_counts = vertex_output['predicted_vertex_counts']
        
        # Predict edges dynamically based on actual vertex counts
        edge_probs_list = []
        edge_indices_list = []
        
        for i in range(batch_size):
            vertex_count = actual_counts[i].item()
            sample_vertices = predicted_vertices[i:i+1, :vertex_count, :]  # Only active vertices
            
            if vertex_count > 1:  # Need at least 2 vertices for edges
                sample_edge_probs, sample_edge_indices = self.edge_predictor(sample_vertices)
                edge_probs_list.append(sample_edge_probs[0])  # Remove batch dimension
                edge_indices_list.extend(sample_edge_indices)
            else:
                # Single vertex or no vertices - no edges
                edge_probs_list.append(torch.zeros(0, device=predicted_vertices.device))
                edge_indices_list = []
        
        # For batch processing, we need to pad edge predictions to same length
        max_edges = max([len(ep) for ep in edge_probs_list]) if edge_probs_list else 0
        
        if max_edges > 0:
            padded_edge_probs = torch.zeros(batch_size, max_edges, device=predicted_vertices.device)
            for i, edge_probs in enumerate(edge_probs_list):
                if len(edge_probs) > 0:
                    padded_edge_probs[i, :len(edge_probs)] = edge_probs
        else:
            padded_edge_probs = torch.zeros(batch_size, 0, device=predicted_vertices.device)
        
        return {
            'vertices': predicted_vertices,
            'edge_probs': padded_edge_probs,
            'edge_indices': edge_indices_list[:max_edges] if max_edges > 0 else [],
            'global_features': global_features,
            'vertex_count_probs': vertex_output['vertex_count_probs'],
            'predicted_vertex_counts': vertex_output['predicted_vertex_counts'],
            'actual_vertex_counts': vertex_output['actual_vertex_counts']
        }



import torch
import torch.nn as nn
# Import specialized sub-modules for 3D wireframe prediction
from models.EdgePredictor import EdgePredictor  # Predicts connections between vertices
from models.PointNetEncoder import PointNetEncoder  # Extracts features from point clouds
from models.VertexPredictor import VertexPredictor  # Predicts vertex positions


class PointCloudToWireframe(nn.Module):
    """
    Complete end-to-end model for point cloud to wireframe prediction
    
    This model takes a 3D point cloud as input and predicts:
    1. 3D coordinates of vertices (fixed number)
    2. Probability of edges between vertex pairs
    """
    
    def __init__(self, input_dim=8, max_vertices=64):
        """
        Initialize the complete wireframe prediction model
        
        Args:
            input_dim (int): Dimension of input point features (default: 8 for X,Y,Z,R,G,B,A,Intensity)
            max_vertices (int): Fixed number of vertices to predict (default: 64)
        """
        super(PointCloudToWireframe, self).__init__()
        
        self.max_vertices = max_vertices
        
        # Point cloud encoder - converts raw point cloud to global and local features
        self.encoder = PointNetEncoder(input_dim=input_dim)
        
        # Vertex predictor - predicts fixed number of vertices
        self.vertex_predictor = VertexPredictor(
            global_feature_dim=512,  # Must match encoder output dimension
            max_vertices=max_vertices
        )
        
        # Edge predictor - predicts connections between vertices using attention
        self.edge_predictor = EdgePredictor(vertex_dim=3)  # 3D vertex coordinates
        
    def forward(self, point_cloud, target_vertex_counts=None):
        """
        Forward pass of the complete wireframe prediction pipeline
        
        Args:
            point_cloud (torch.Tensor): Input point cloud (batch_size, num_points, input_dim)
            target_vertex_counts (torch.Tensor, optional): Kept for compatibility but ignored
            
        Returns:
            dict: Dictionary containing:
                - vertices: Predicted vertex coordinates (batch_size, max_vertices, 3)
                - edge_probs: Edge existence probabilities (batch_size, num_edges)
                - edge_indices: List of edge index pairs
                - global_features: Encoded global features
                - actual_vertex_counts: Fixed vertex counts (all max_vertices)
        """
        
        # Encode point cloud
        global_features, point_features = self.encoder(point_cloud)
        
        # Predict vertices (fixed count)
        vertex_output = self.vertex_predictor(global_features, target_vertex_counts)
        predicted_vertices = vertex_output['vertices']  # (batch_size, max_vertices, 3)
        
        # Predict edges for all vertices
        batch_size = predicted_vertices.shape[0]
        edge_probs_list = []
        edge_indices_list = []
        
        # Use actual vertex counts from target for edge prediction during training
        if self.training and target_vertex_counts is not None:
            # During training, use ground truth vertex counts for edge prediction
            for i in range(batch_size):
                actual_count = target_vertex_counts[i].item()
                sample_vertices = predicted_vertices[i:i+1, :actual_count, :]  # Only actual vertices
                
                # Predict edges for actual vertex count
                sample_edge_probs, sample_edge_indices = self.edge_predictor(sample_vertices)
                edge_probs_list.append(sample_edge_probs[0])
                if i == 0:  # Store edge indices once (same pattern for all samples with same count)
                    edge_indices_list = sample_edge_indices
        else:
            # During inference, use all vertices
            for i in range(batch_size):
                sample_vertices = predicted_vertices[i:i+1, :, :]  # All vertices
                
                # Predict edges
                sample_edge_probs, sample_edge_indices = self.edge_predictor(sample_vertices)
                edge_probs_list.append(sample_edge_probs[0])
                if i == 0:  # Store edge indices once
                    edge_indices_list = sample_edge_indices
        
        # Pad edge predictions for batch processing
        max_edges = max([len(ep) for ep in edge_probs_list]) if edge_probs_list else 0
        
        if max_edges > 0:
            # Create padded tensor for different edge counts per sample
            padded_edge_probs = torch.zeros(batch_size, max_edges, device=predicted_vertices.device)
            for i, edge_probs in enumerate(edge_probs_list):
                if len(edge_probs) > 0:
                    padded_edge_probs[i, :len(edge_probs)] = edge_probs
        else:
            padded_edge_probs = torch.zeros(batch_size, 0, device=predicted_vertices.device)
        
        return {
            'vertices': predicted_vertices,
            'edge_probs': padded_edge_probs,
            'edge_indices': edge_indices_list,
            'global_features': global_features,
            'actual_vertex_counts': vertex_output['actual_vertex_counts']  # All equal to max_vertices
        }


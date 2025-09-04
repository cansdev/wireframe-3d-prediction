
import torch
import torch.nn as nn
# Import specialized sub-modules for 3D wireframe prediction
from models.EdgePredictor import EdgePredictor  # Predicts connections between vertices
from models.PointNetEncoder import PointNetEncoder  # Extracts features from point clouds
from models.VertexPredictor import VertexPredictor  # Predicts vertex positions and counts


class PointCloudToWireframe(nn.Module):
    """
    Complete end-to-end model for point cloud to wireframe prediction
    
    This model takes a 3D point cloud as input and predicts:
    1. Number of vertices in the wireframe
    2. 3D coordinates of each vertex
    3. Probability of edges between vertex pairs
    
    The model uses dynamic vertex count prediction to handle variable-sized wireframes.
    """
    
    def __init__(self, input_dim=8, max_vertices=64):
        """
        Initialize the complete wireframe prediction model
        
        Args:
            input_dim (int): Dimension of input point features (default: 8 for X,Y,Z,R,G,B,A,Intensity)
            max_vertices (int): Maximum number of vertices the model can predict (default: 64)
        """
        super(PointCloudToWireframe, self).__init__()
        
        self.max_vertices = max_vertices
        
        # Point cloud encoder - converts raw point cloud to global and local features
        self.encoder = PointNetEncoder(input_dim=input_dim)
        
        # Dynamic vertex predictor - predicts vertex count and coordinates
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
            point_cloud (torch.Tensor): Input point cloud of shape (batch_size, num_points, input_dim)
            target_vertex_counts (torch.Tensor, optional): Ground truth vertex counts during training
            
        Returns:
            dict: Dictionary containing:
                - vertices: Predicted vertex coordinates (batch_size, max_vertices, 3)
                - edge_probs: Edge existence probabilities (batch_size, num_edges)
                - edge_indices: List of edge index pairs
                - global_features: Encoded global features from point cloud
                - vertex_count_probs: Probability distribution over vertex counts
                - predicted_vertex_counts: Predicted number of vertices per sample
                - actual_vertex_counts: Actual vertex counts used (training or predicted)
        """
        
        # STEP 1: Encode point cloud into global and local features
        # global_features: (batch_size, 512) - overall shape representation
        # point_features: (batch_size, num_points, 512) - per-point features
        global_features, point_features = self.encoder(point_cloud)
        
        # STEP 2: Predict vertices (count + coordinates) from global features
        # Note: point_features could be integrated here for better vertex prediction
        vertex_output = self.vertex_predictor(global_features, target_vertex_counts) # point_features eklenmesi mümkün $$$$$$$$$
        predicted_vertices = vertex_output['vertices']  # Shape: (batch_size, max_vertices, 3)
        
        # STEP 3: Apply training constraints (mask out excess vertices during training)
        # During training, apply hard constraint on vertex count
        if self.training and target_vertex_counts is not None:
            # Mask out vertices beyond target count to enforce ground truth structure
            batch_size = predicted_vertices.shape[0]
            for i in range(batch_size):
                count = target_vertex_counts[i].item()
                if count < self.max_vertices:
                    # Zero out vertices beyond the target count
                    predicted_vertices[i, count:, :] = 0.0
        
        # STEP 4: Determine actual vertex counts for edge prediction
        # For edge prediction, we need to mask out unused vertices
        batch_size = predicted_vertices.shape[0] # (batch_size, max_vertices, 3)
        
        # Get actual vertex counts for each sample in batch
        if self.training and target_vertex_counts is not None:
            actual_counts = target_vertex_counts  # Use ground truth during training
        else:
            actual_counts = vertex_output['predicted_vertex_counts']  # Use model predictions during inference
        
        # STEP 5: Predict edges dynamically for each sample based on actual vertex count
        # Predict edges dynamically based on actual vertex counts
        edge_probs_list = []  # Store edge probabilities for each sample
        edge_indices_list = []  # Store edge indices for each sample
        
        # Process each sample individually due to variable vertex counts
        for i in range(batch_size):
            vertex_count = actual_counts[i].item()
            sample_vertices = predicted_vertices[i:i+1, :vertex_count, :]  # Only active vertices
            
            if vertex_count > 1:  # Need at least 2 vertices for edges
                # Predict edge probabilities and indices using attention mechanism
                sample_edge_probs, sample_edge_indices = self.edge_predictor(sample_vertices)
                edge_probs_list.append(sample_edge_probs[0])  # Remove batch dimension
                edge_indices_list.extend(sample_edge_indices)
            else:
                # Single vertex or no vertices - no edges possible
                edge_probs_list.append(torch.zeros(0, device=predicted_vertices.device))
                edge_indices_list = []
        
        # STEP 6: Pad edge predictions for batch processing
        # For batch processing, we need to pad edge predictions to same length
        max_edges = max([len(ep) for ep in edge_probs_list]) if edge_probs_list else 0
        
        if max_edges > 0:
            # Create padded tensor for batch processing
            padded_edge_probs = torch.zeros(batch_size, max_edges, device=predicted_vertices.device)
            for i, edge_probs in enumerate(edge_probs_list):
                if len(edge_probs) > 0:
                    padded_edge_probs[i, :len(edge_probs)] = edge_probs
        else:
            # No edges predicted for any sample
            padded_edge_probs = torch.zeros(batch_size, 0, device=predicted_vertices.device)
        
        # STEP 7: Return complete prediction results
        return {
            'vertices': predicted_vertices,  # (batch_size, max_vertices, 3) - predicted 3D coordinates
            'edge_probs': padded_edge_probs,  # (batch_size, max_edges) - edge existence probabilities
            'edge_indices': edge_indices_list[:max_edges] if max_edges > 0 else [],  # List of (i,j) edge pairs
            'global_features': global_features,  # (batch_size, 512) - encoded global features
            'vertex_count_probs': vertex_output['vertex_count_probs'],  # (batch_size, max_vertices+1) - count distribution
            'predicted_vertex_counts': vertex_output['predicted_vertex_counts'],  # (batch_size,) - predicted counts
            'actual_vertex_counts': vertex_output['actual_vertex_counts']  # (batch_size,) - actual counts used
        }


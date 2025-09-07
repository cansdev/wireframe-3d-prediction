
import torch
import torch.nn as nn

class VertexPredictor(nn.Module):
    """
    Vertex predictor for wireframe reconstruction
    
    This module predicts 3D vertex coordinates from global point cloud features.
    The number of vertices is now fixed based on max_vertices parameter.
    """
    
    def __init__(self, global_feature_dim=512, max_vertices=64, vertex_dim=4):
        """
        Initialize the vertex predictor
        
        Args:
            global_feature_dim (int): Dimension of input global features from encoder (default: 512)
            max_vertices (int): Maximum number of vertices to predict (default: 64)
            vertex_dim (int): Dimension of vertex features (default: 4 for X,Y,Z,existence_probability)
        """
        super(VertexPredictor, self).__init__()
        
        self.max_vertices = max_vertices
        self.vertex_dim = vertex_dim
        
        # Deep MLPs for vertex coordinate prediction
        self.vertex_mlp1 = nn.Sequential(
            nn.Linear(global_feature_dim, 4096),  # Input: global features only
            nn.LayerNorm(4096),                   # Layer normalization
            nn.ReLU(inplace=True),                # Activation
            nn.Dropout(0.0),                      # No dropout for coordinate prediction
        )
        
        self.vertex_mlp2 = nn.Sequential(
            nn.Linear(4096, 2048),              # Compression layer
            nn.LayerNorm(2048),                 # Normalization
            nn.ReLU(inplace=True),              # Activation
            nn.Dropout(0.0),                    # No dropout
        )
        
        self.vertex_mlp3 = nn.Sequential(
            nn.Linear(2048, 2048),              # Maintain dimension for deep processing
            nn.LayerNorm(2048),                 # Normalization
            nn.ReLU(inplace=True),              # Activation
            nn.Dropout(0.0),                    # No dropout
        )
        
        self.vertex_mlp4 = nn.Sequential(
            nn.Linear(2048, 1024),              # Further compression
            nn.LayerNorm(1024),                 # Normalization
            nn.ReLU(inplace=True),              # Activation
            nn.Dropout(0.0),                    # No dropout
        )
        
        # Final output layer for all vertex features (coordinates + existence probability)
        self.final_layer = nn.Linear(1024, max_vertices * vertex_dim)  # Output: flattened vertex features
        
        # Residual projection layers for better gradient flow
        self.residual_proj1 = nn.Linear(global_feature_dim, 2048)  # First residual projection
        self.residual_proj2 = nn.Linear(global_feature_dim, 1024)  # Second residual projection
        
    def forward(self, global_features, target_vertex_counts=None):
        """
        Forward pass for vertex prediction
        
        Args:
            global_features (torch.Tensor): Global features from point cloud encoder (batch_size, global_feature_dim)
            target_vertex_counts (torch.Tensor, optional): Ground truth vertex counts for dynamic prediction
            
        Returns:
            dict: Dictionary containing:
                - vertices: Predicted vertex coordinates (batch_size, max_vertices, 3)
                - existence_probabilities: Vertex existence probabilities (batch_size, max_vertices)
                - actual_vertex_counts: Dynamic vertex counts based on existence probabilities
        """
        batch_size = global_features.shape[0]
        
        # Deep MLP processing for coordinate prediction with residual connections
        x = self.vertex_mlp1(global_features)    # First MLP layer
        x = self.vertex_mlp2(x)                  # Second MLP layer
        
        # First residual connection
        residual1 = self.residual_proj1(global_features)
        x = self.vertex_mlp3(x) + residual1      # Add residual and continue processing
        
        # Second residual connection
        residual2 = self.residual_proj2(global_features)
        x = self.vertex_mlp4(x) + residual2      # Add second residual
        
        # Generate final vertex features (coordinates + existence probability)
        vertex_features = self.final_layer(x)      # (batch_size, max_vertices * vertex_dim)
        vertex_features = vertex_features.view(batch_size, self.max_vertices, self.vertex_dim)  # Reshape to vertex features
        
        # Split into coordinates and existence probabilities
        vertex_coords = vertex_features[:, :, :3]  # First 3 dimensions are coordinates (x, y, z)
        existence_logits = vertex_features[:, :, 3]  # 4th dimension is existence logit
        existence_probabilities = torch.sigmoid(existence_logits)  # Convert to probabilities [0, 1]
        
        # Calculate dynamic vertex counts based on existence probabilities
        # Use threshold of 0.5 to determine if vertex exists
        vertex_exists = existence_probabilities > 0.5  # (batch_size, max_vertices)
        actual_vertex_counts = vertex_exists.sum(dim=1)  # (batch_size,)
        
        return {
            'vertices': vertex_coords,                           # Predicted 3D coordinates (batch_size, max_vertices, 3)
            'existence_probabilities': existence_probabilities,  # Vertex existence probabilities (batch_size, max_vertices)
            'actual_vertex_counts': actual_vertex_counts         # Dynamic vertex counts based on probabilities
        }
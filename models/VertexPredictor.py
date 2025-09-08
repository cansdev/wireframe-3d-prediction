
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
        
    def forward(self, global_features, point_features, target_vertex_counts=None):
        """
        Forward pass for vertex prediction, now using point_features to condition global features.

        Args:
            global_features (torch.Tensor): Global features from point cloud encoder (batch_size, global_feature_dim)
            point_features (torch.Tensor): Per-point features (batch_size, num_points, point_feature_dim)
            target_vertex_counts (torch.Tensor, optional): Ground truth vertex counts for dynamic prediction

        Returns:
            dict: {
            'vertices': (batch_size, max_vertices, 3),
            'existence_probabilities': (batch_size, max_vertices),
            'actual_vertex_counts': (batch_size,)
            }
        """
        batch_size = global_features.shape[0]

        # If point features are provided, pool them (mean + max) and project to global_feature_dim,
        # then fuse with global_features (residual addition). The projection layer is created lazily
        # so its parameters are registered on first forward.
        if point_features is not None:
            # point_features: (B, N, C)
            pooled_mean = point_features.mean(dim=1)                    # (B, C)
            pooled_max = point_features.max(dim=1).values               # (B, C)
            pooled = torch.cat([pooled_mean, pooled_max], dim=1)        # (B, 2C)

            # Determine target global dim from an existing layer (registered in __init__)
            global_dim = self.residual_proj1.in_features

            # Lazily create projection from pooled point-features -> global_dim
            if not hasattr(self, "point_pool_proj"):
                self.point_pool_proj = nn.Linear(pooled.shape[1], global_dim)
                # Move the new layer to the same device as the input tensor
                self.point_pool_proj = self.point_pool_proj.to(pooled.device)

            point_global = self.point_pool_proj(pooled)                 # (B, global_dim)
            enhanced_global = global_features + point_global            # Fuse via residual addition
        else:
            enhanced_global = global_features

        # Deep MLP processing for coordinate prediction with residual connections
        x = self.vertex_mlp1(enhanced_global)    # First MLP layer
        x = self.vertex_mlp2(x)                  # Second MLP layer

        # First residual connection (project enhanced global into MLP space)
        residual1 = self.residual_proj1(enhanced_global)
        x = self.vertex_mlp3(x) + residual1      # Add residual and continue processing

        # Second residual connection
        residual2 = self.residual_proj2(enhanced_global)
        x = self.vertex_mlp4(x) + residual2      # Add second residual

        # Generate final vertex features (coordinates + existence probability)
        vertex_features = self.final_layer(x)      # (batch_size, max_vertices * vertex_dim)
        vertex_features = vertex_features.view(batch_size, self.max_vertices, self.vertex_dim)

        # Split into coordinates and existence probabilities
        vertex_coords = vertex_features[:, :, :3]                    # (B, max_vertices, 3)
        existence_logits = vertex_features[:, :, 3]                  # (B, max_vertices)
        existence_probabilities = torch.sigmoid(existence_logits)    # (B, max_vertices)

        # Calculate dynamic vertex counts based on existence probabilities (threshold=0.5)
        vertex_exists = existence_probabilities > 0.5                # (B, max_vertices)
        actual_vertex_counts = vertex_exists.sum(dim=1)              # (B,)

        return {
            'vertices': vertex_coords,
            'existence_probabilities': existence_probabilities,
            'actual_vertex_counts': actual_vertex_counts
        }
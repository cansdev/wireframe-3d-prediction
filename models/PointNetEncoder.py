
import torch
import torch.nn as nn

class PointNetEncoder(nn.Module):
    """
    PointNet-based encoder for extracting features from 3D point clouds
    
    This encoder processes variable-sized point clouds and extracts both:
    1. Global features representing the overall shape
    2. Point-wise features for each individual point
    
    The architecture uses:
    - Multi-layer perceptron (MLP) applied to each point independently
    - Dual pooling (max + average) for robust global feature extraction
    - Mask-aware processing to handle padded/variable-sized inputs
    """
    
    def __init__(self, input_dim=8, hidden_dims=[512, 1024, 2048, 1024], output_dim=512):
        """
        Initialize the PointNet encoder
        
        Args:
            input_dim (int): Dimension of input point features (default: 8 for X,Y,Z,R,G,B,A,Intensity)
            hidden_dims (list): Hidden layer dimensions for the point-wise MLP
            output_dim (int): Final output dimension for global features
        """
        super(PointNetEncoder, self).__init__()
        
        # Build point-wise MLP layers
        layers = []
        prev_dim = input_dim
        
        # Create deep MLP for point feature extraction
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),  # Linear transformation
                nn.LayerNorm(hidden_dim),         # Normalization for stable training
                nn.ReLU(inplace=True),            # Non-linear activation
                nn.Dropout(0.0)                   # Dropout disabled for overfitting capability
            ])
            prev_dim = hidden_dim
            
        # Final projection layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Point-wise feature extractor (applied to each point independently)
        self.mlp = nn.Sequential(*layers)   
        
        # Global pooling operations for aggregating point features
        # Enhanced pooling - combine max and mean for robust feature aggregation
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)  # Extracts maximum response
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Extracts average response
        
        # Feature fusion network to combine max and average pooled features
        # Enhanced feature fusion with more capacity for batch training
        self.feature_fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim * 4),  # Expand dimension for rich representation
            nn.LayerNorm(output_dim * 4),               # Normalization
            nn.ReLU(inplace=True),                      # Activation
            nn.Linear(output_dim * 4, output_dim * 2),  # Intermediate compression
            nn.LayerNorm(output_dim * 2),               # Normalization
            nn.ReLU(inplace=True),                      # Activation
            nn.Linear(output_dim * 2, output_dim)       # Final compression to target dimension
        )
        
    def forward(self, x):
        """
        Forward pass of the PointNet encoder
        
        Args:
            x (torch.Tensor): Input point cloud of shape (batch_size, num_points, input_dim)
            
        Returns:
            tuple: (global_features, point_features)
                - global_features (torch.Tensor): Global shape representation (batch_size, output_dim)
                - point_features (torch.Tensor): Per-point features (batch_size, num_points, output_dim)
        """
        
        # Input shape: (batch_size, num_points, input_dim)
        batch_size, num_points, input_dim = x.shape

        # STEP 1: Create mask to identify valid (non-padded) points
        # Build mask of valid points from detached input to avoid autograd overhead
        mask = (x.detach().abs().sum(dim=-1) > 1e-9)  # (batch_size, num_points) - True for valid points
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1).float()  # (batch_size, 1) - count valid points

        # STEP 2: Reshape input for point-wise MLP processing
        # Reshape for MLP processing (PointNet processes each point independently)
        x = x.view(-1, input_dim)  # (batch_size * num_points, input_dim) - flatten batch and point dimensions

        # STEP 3: Apply point-wise MLP to extract features from each point
        # Apply MLP to each point independently (core PointNet operation)
        point_features = self.mlp(x)  # (batch_size * num_points, output_dim)

        # STEP 4: Reshape back to original batch structure
        # Reshape back to separate batch and point dimensions
        point_features = point_features.view(batch_size, num_points, -1)  # (batch_size, num_points, output_dim)

        # STEP 5: Global feature aggregation using mask-aware pooling
        # Mask-aware global pooling to ignore zero-padded points
        # Masked average pooling over valid points
        masked_features = point_features * mask.unsqueeze(-1)  # (batch_size, num_points, feat_dim) - zero out invalid points
        avg_features = masked_features.sum(dim=1) / valid_counts  # (batch_size, feat_dim) - compute mean over valid points

        # Masked max pooling: set invalid positions to -inf before max operation
        masked_for_max = point_features.masked_fill(~mask.unsqueeze(-1), float('-inf'))  # Set invalid to -inf
        # Use built-in max reduction for computational efficiency
        max_features, _ = masked_for_max.max(dim=1)  # (batch_size, feat_dim) - extract maximum over valid points
        # Safety check for numerical stability (shouldn't happen but keep robust)
        max_features = torch.where(torch.isfinite(max_features), max_features, torch.zeros_like(max_features))

        # STEP 6: Combine pooled features and apply fusion network
        # Combine max and average features for rich global representation
        combined_features = torch.cat([max_features, avg_features], dim=1)  # (batch_size, 2*output_dim)
        global_features = self.feature_fusion(combined_features)  # (batch_size, output_dim)

        # Return both global and point-wise features
        return global_features, point_features
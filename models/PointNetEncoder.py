
import torch
import torch.nn as nn

class PointNetEncoder(nn.Module):
    """Massive PointNet encoder for total overfitting"""
    
    def __init__(self, input_dim=8, hidden_dims=[512, 1024, 2048, 1024], output_dim=512):  # Increased capacity for batch training
        super(PointNetEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.0)  # No dropout for total overfitting
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)   
        
        # Enhanced pooling - combine max and mean
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Enhanced feature fusion with more capacity for batch training
        self.feature_fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim * 4),  # Increased capacity
            nn.LayerNorm(output_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim * 4, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim * 2, output_dim)
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_points, input_dim)
        batch_size, num_points, input_dim = x.shape

        # Build mask of valid points from detached input to avoid autograd overhead
        mask = (x.detach().abs().sum(dim=-1) > 1e-9)  # (batch_size, num_points)
        valid_counts = mask.sum(dim=1, keepdim=True).clamp(min=1).float()  # (batch_size, 1)

        # Reshape for MLP processing
        main
        x = x.view(-1, input_dim)  # (batch_size * num_points, input_dim) reshape kullanılabilir mi? $$$$$$$$$$$$


        
        # Apply MLP to each point
        point_features = self.mlp(x)  # (batch_size * num_points, output_dim)

        # Reshape back
        point_features = point_features.view(batch_size, num_points, -1)

        # Mask-aware global pooling to ignore zero-padded points
        # Masked average pooling over valid points
        masked_features = point_features * mask.unsqueeze(-1)  # (batch_size, num_points, feat_dim)
        avg_features = masked_features.sum(dim=1) / valid_counts  # (batch_size, feat_dim)

        # Masked max pooling: set invalid positions to -inf before max
        masked_for_max = point_features.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        # Use built-in max reduction for speed
        max_features, _ = masked_for_max.max(dim=1)  # (batch_size, feat_dim)
        # Safety for all-invalid (shouldn't happen but keep numerically robust)
        max_features = torch.where(torch.isfinite(max_features), max_features, torch.zeros_like(max_features))

        # Combine features with more processing
        combined_features = torch.cat([max_features, avg_features], dim=1)
        global_features = self.feature_fusion(combined_features)

        return global_features, point_features
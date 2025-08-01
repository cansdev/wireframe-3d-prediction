
import torch
import torch.nn as nn

class PointNetEncoder(nn.Module):
    """PointNet-inspired encoder for point cloud features"""
    
    def __init__(self, input_dim=8, hidden_dims=[64, 128, 256], output_dim=512):
        super(PointNetEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)   # kaldirilabilir
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x):
        # x shape: (batch_size, num_points, input_dim)
        batch_size, num_points, input_dim = x.shape
        
        # Reshape for MLP processing
        x = x.view(-1, input_dim)  # (batch_size * num_points, input_dim)
        
        # Apply MLP to each point
        point_features = self.mlp(x)  # (batch_size * num_points, output_dim)
        
        # Reshape back
        point_features = point_features.view(batch_size, num_points, -1)
        
        # Global max pooling across points
        # Transpose for pooling: (batch_size, output_dim, num_points)
        point_features = point_features.transpose(1, 2)
        global_features = self.global_pool(point_features).squeeze(-1)
        
        return global_features, point_features.transpose(1, 2)


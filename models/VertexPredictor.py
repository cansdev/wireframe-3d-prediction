
import torch.nn as nn

class VertexPredictor(nn.Module):
    """Predict vertex locations from global point cloud features"""
    
    def __init__(self, global_feature_dim=512, num_vertices=32, vertex_dim=3):
        super(VertexPredictor, self).__init__()
        
        self.num_vertices = num_vertices
        self.vertex_dim = vertex_dim
        
        self.vertex_mlp = nn.Sequential(
            nn.Linear(global_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_vertices * vertex_dim)
        )
        
    def forward(self, global_features):
        # Predict vertex coordinates
        vertex_coords = self.vertex_mlp(global_features)
        vertex_coords = vertex_coords.view(-1, self.num_vertices, self.vertex_dim)
        return vertex_coords


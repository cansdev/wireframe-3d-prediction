
import torch
import torch.nn as nn

class VertexPredictor(nn.Module):
    """Dynamic vertex predictor that can handle variable vertex counts"""
    
    def __init__(self, global_feature_dim=512, max_vertices=64, vertex_dim=3):
        super(VertexPredictor, self).__init__()
        
        self.max_vertices = max_vertices
        self.vertex_dim = vertex_dim
        
        # First predict the number of vertices
        self.vertex_count_predictor = nn.Sequential(
            nn.Linear(global_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, max_vertices),  # Probability distribution over possible vertex counts
            nn.Softmax(dim=1)
        )
        
        # Enhanced vertex coordinate predictor
        self.vertex_mlp1 = nn.Sequential(
            nn.Linear(global_feature_dim + 1, 2048),  # +1 for vertex count info
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),  # No dropout for overfitting
        )
        
        self.vertex_mlp2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
        )
        
        self.vertex_mlp3 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
        )
        
        self.vertex_mlp4 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
        )
        
        # Dynamic output layer - predicts coordinates for maximum possible vertices
        self.final_layer = nn.Linear(512, max_vertices * vertex_dim)
        
        # Residual connections
        self.residual_proj1 = nn.Linear(global_feature_dim + 1, 1024)
        self.residual_proj2 = nn.Linear(global_feature_dim + 1, 512)
        
    def forward(self, global_features, target_vertex_counts=None):
        batch_size = global_features.shape[0]
        
        # Predict vertex counts
        vertex_count_probs = self.vertex_count_predictor(global_features)
        predicted_vertex_counts = torch.argmax(vertex_count_probs, dim=1) + 1  # +1 because we start from 1 vertex
        
        # Use target counts during training, predicted counts during inference
        if self.training and target_vertex_counts is not None:
            actual_vertex_counts = target_vertex_counts.float()
        else:
            actual_vertex_counts = predicted_vertex_counts.float()
        
        # Normalize vertex counts to [0, 1] range for network input
        normalized_counts = actual_vertex_counts.unsqueeze(1) / self.max_vertices
        
        # Concatenate global features with vertex count information
        enhanced_features = torch.cat([global_features, normalized_counts], dim=1)
        
        # Enhanced forward pass with vertex count conditioning
        x = self.vertex_mlp1(enhanced_features)
        x = self.vertex_mlp2(x)
        
        # First residual connection
        residual1 = self.residual_proj1(enhanced_features)
        x = self.vertex_mlp3(x) + residual1
        
        # Second residual connection
        residual2 = self.residual_proj2(enhanced_features)
        x = self.vertex_mlp4(x) + residual2
        
        # Final prediction
        vertex_coords = self.final_layer(x)
        vertex_coords = vertex_coords.view(batch_size, self.max_vertices, self.vertex_dim)
        
        return {
            'vertices': vertex_coords,
            'vertex_count_probs': vertex_count_probs,
            'predicted_vertex_counts': predicted_vertex_counts,
            'actual_vertex_counts': actual_vertex_counts.long()
        }
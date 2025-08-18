
import torch
import torch.nn as nn

class VertexPredictor(nn.Module):
    """Dynamic vertex predictor that can handle variable vertex counts"""
    
    def __init__(self, global_feature_dim=512, max_vertices=64, vertex_dim=3):
        super(VertexPredictor, self).__init__()
        
        self.max_vertices = max_vertices
        self.vertex_dim = vertex_dim
        
        # First predict the number of vertices with stronger regularization
        self.vertex_count_predictor = nn.Sequential(
            nn.Linear(global_feature_dim, 512),  # Increased capacity for batch training
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(512, 256),  # Increased capacity
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Added dropout
            nn.Linear(256, max_vertices + 1)  # Predict counts from 0 to max_vertices (0-indexed)
        )
        
        # Enhanced vertex coordinate predictor with increased capacity
        self.vertex_mlp1 = nn.Sequential(
            nn.Linear(global_feature_dim + 1, 4096),  # Increased from 2048 for batch training
            nn.LayerNorm(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),  # No dropout for overfitting
        )
        
        self.vertex_mlp2 = nn.Sequential(
            nn.Linear(4096, 2048),  # Increased from 1024
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
        )
        
        self.vertex_mlp3 = nn.Sequential(
            nn.Linear(2048, 2048),  # Increased from 1024
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
        )
        
        self.vertex_mlp4 = nn.Sequential(
            nn.Linear(2048, 1024),  # Increased from 512
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
        )
        
        # Dynamic output layer - predicts coordinates for maximum possible vertices
        self.final_layer = nn.Linear(1024, max_vertices * vertex_dim)  # Input increased from 512
        
        # Residual connections with increased capacity
        self.residual_proj1 = nn.Linear(global_feature_dim + 1, 2048)  # Increased from 1024
        self.residual_proj2 = nn.Linear(global_feature_dim + 1, 1024)  # Increased from 512
        
    def forward(self, global_features, target_vertex_counts=None):
        batch_size = global_features.shape[0]
        
        # Predict vertex counts with temperature scaling
        vertex_count_logits = self.vertex_count_predictor(global_features)
        temperature = 0.1 if not self.training else 0.2  # Even lower temperature for very sharp predictions during inference
        vertex_count_probs = torch.softmax(vertex_count_logits / temperature, dim=-1)
        
        # More aggressive argmax selection
        predicted_vertex_counts = torch.argmax(vertex_count_probs, dim=1)  # 0-indexed, represents actual count
        
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
        
        # During inference, explicitly zero out vertices beyond predicted count
        if not self.training:
            for i in range(batch_size):
                count = predicted_vertex_counts[i].item()
                if count < self.max_vertices:
                    vertex_coords[i, count:, :] = 0.0  # Zero out unused vertices
        
        return {
            'vertices': vertex_coords,
            'vertex_count_probs': vertex_count_probs,
            'predicted_vertex_counts': predicted_vertex_counts,
            'actual_vertex_counts': actual_vertex_counts.long()
        }
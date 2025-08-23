import torch
import torch.nn as nn

class EdgePredictor(nn.Module):
    """Alternative version with attention mechanism for better feature learning"""
    
    def __init__(self, vertex_dim=3, hidden_dim=512, num_heads=8):
        super(EdgePredictor, self).__init__()
        
        # Project vertices to higher dimension
        self.vertex_proj = nn.Linear(vertex_dim, hidden_dim)
        
        # Self-attention to capture vertex relationships
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Edge prediction head
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.register_buffer('edge_indices_cache', None)
        self.cached_num_vertices = None
        
    def _get_edge_indices(self, num_vertices):
        if self.cached_num_vertices != num_vertices or self.edge_indices_cache is None:
            indices = []
            for i in range(num_vertices):
                for j in range(i + 1, num_vertices):
                    indices.append([i, j])
            self.edge_indices_cache = torch.tensor(indices, dtype=torch.long, device=next(self.parameters()).device)
            self.cached_num_vertices = num_vertices
        return self.edge_indices_cache
    
    def forward(self, vertices):
        batch_size, num_vertices, vertex_dim = vertices.shape
        
        # Project vertices to hidden dimension
        vertex_features = self.vertex_proj(vertices)  # (batch_size, num_vertices, hidden_dim)
        
        # Apply self-attention
        attended_features, _ = self.attention(
            vertex_features, vertex_features, vertex_features
        )  # (batch_size, num_vertices, hidden_dim)
        
        # Residual connection
        vertex_features = vertex_features + attended_features
        
        # Get edge indices
        edge_indices = self._get_edge_indices(num_vertices)
        i_indices = edge_indices[:, 0]
        j_indices = edge_indices[:, 1]
        
        # Gather vertex pairs
        v1 = vertex_features[:, i_indices, :]  # (batch_size, num_edges, hidden_dim)
        v2 = vertex_features[:, j_indices, :]  # (batch_size, num_edges, hidden_dim)
        
        # Concatenate and predict
        edge_features = torch.cat([v1, v2], dim=-1)  # (batch_size, num_edges, hidden_dim * 2)
        edge_features_flat = edge_features.view(-1, edge_features.shape[-1])
        
        edge_logits = self.edge_mlp(edge_features_flat)
        edge_probs = torch.sigmoid(edge_logits).view(batch_size, -1)
        
        return edge_probs, edge_indices.tolist()
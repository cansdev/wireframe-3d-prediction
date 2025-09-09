import torch
import torch.nn as nn

class EdgePredictor(nn.Module):
    """
    Attention-based edge predictor for wireframe construction
    
    This module predicts the probability of edges existing between vertex pairs in a wireframe.
    It uses self-attention to capture vertex relationships and predict connectivity.
    
    Architecture:
    1. Project vertices to higher-dimensional space
    2. Apply multi-head self-attention to capture vertex interactions
    3. For each vertex pair, concatenate features and predict edge probability
    
    The model handles variable number of vertices by dynamically generating edge indices.
    """
    
    def __init__(self, vertex_dim=3, hidden_dim=512, num_heads=8):
        """
        Initialize the attention-based edge predictor
        
        Args:
            vertex_dim (int): Dimension of vertex coordinates (default: 3 for X,Y,Z)
            hidden_dim (int): Hidden dimension for feature processing (default: 512)
            num_heads (int): Number of attention heads (default: 8)
        """
        super(EdgePredictor, self).__init__()
        
        # Enhanced vertex feature extraction
        self.vertex_proj = nn.Sequential(
            nn.Linear(vertex_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Multi-head self-attention to capture vertex relationships and interactions
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,      # Feature dimension
            num_heads=num_heads,       # Number of parallel attention heads
            dropout=0.1,               # Attention dropout for regularization
            batch_first=True           # Input format: (batch_size, seq_len, features)
        )
        
        # Additional spatial encoding for better geometric understanding
        self.spatial_proj = nn.Sequential(
            nn.Linear(vertex_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )
        
        # Enhanced edge prediction MLP with spatial features
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + vertex_dim * 2 + 1, hidden_dim),  # Features + coordinates + distance
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def _get_edge_indices(self, num_vertices):
        """
        Generate all possible edge indices for a given number of vertices
        
        For N vertices, generates all unique pairs (i,j) where i < j.
        
        Args:
            num_vertices (int): Number of vertices in the wireframe
            
        Returns:
            torch.Tensor: Edge indices of shape (num_edges, 2) where num_edges = N*(N-1)/2
        """
        # Generate all unique vertex pairs (combinatorial approach)
        indices = []
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):  # Only upper triangle (i < j)
                indices.append([i, j])
        
        # ✅ Generate fresh indices each time - no caching
        return torch.tensor(indices, dtype=torch.long, device=next(self.parameters()).device)
    
    def forward(self, vertices):
        """
        Forward pass for edge prediction
        
        Args:
            vertices (torch.Tensor): Vertex coordinates of shape (batch_size, num_vertices, vertex_dim)
            
        Returns:
            tuple: (edge_probs, edge_indices)
                - edge_probs (torch.Tensor): Edge probabilities of shape (batch_size, num_edges)
                - edge_indices (list): List of [i,j] pairs representing potential edges
        """
        batch_size, num_vertices, vertex_dim = vertices.shape
        
        # STEP 1: Enhanced vertex feature extraction
        vertex_features = self.vertex_proj(vertices)  # (batch_size, num_vertices, hidden_dim)
        
        # STEP 2: Apply self-attention to capture vertex relationships
        attended_features, attention_weights = self.attention(
            vertex_features, vertex_features, vertex_features
        )  # (batch_size, num_vertices, hidden_dim)
        
        # STEP 3: Add residual connection and combine with spatial features
        vertex_features = vertex_features + attended_features
        
        # STEP 4: Generate edge indices for all possible vertex pairs
        edge_indices = self._get_edge_indices(num_vertices)  # (num_edges, 2)
        i_indices = edge_indices[:, 0]  # First vertex of each pair
        j_indices = edge_indices[:, 1]  # Second vertex of each pair
        
        # STEP 5: Gather features and coordinates for each vertex pair
        v1_features = vertex_features[:, i_indices, :]  # (batch_size, num_edges, hidden_dim)
        v2_features = vertex_features[:, j_indices, :]  # (batch_size, num_edges, hidden_dim)
        
        v1_coords = vertices[:, i_indices, :]  # (batch_size, num_edges, vertex_dim)
        v2_coords = vertices[:, j_indices, :]  # (batch_size, num_edges, vertex_dim)
        
        # STEP 6: Predict edge probabilities with enhanced features
        # Calculate spatial distance features
        distances = torch.norm(v1_coords - v2_coords, dim=-1, keepdim=True)  # (batch_size, num_edges, 1)
        
        # Concatenate learned features + raw spatial coordinates + distances
        edge_features = torch.cat([v1_features, v2_features, v1_coords, v2_coords, distances], dim=-1)
        edge_features_flat = edge_features.view(-1, edge_features.shape[-1])
        
        # Apply enhanced edge prediction MLP
        edge_logits = self.edge_mlp(edge_features_flat)
        edge_probs = torch.sigmoid(edge_logits).view(batch_size, -1)
        
        return edge_probs, edge_indices.tolist()
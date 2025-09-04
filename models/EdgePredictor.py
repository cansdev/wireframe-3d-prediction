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
        
        # Project vertices from 3D coordinates to higher-dimensional feature space
        self.vertex_proj = nn.Linear(vertex_dim, hidden_dim)
        
        # Multi-head self-attention to capture vertex relationships and interactions
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,      # Feature dimension
            num_heads=num_heads,       # Number of parallel attention heads
            dropout=0.1,               # Attention dropout for regularization
            batch_first=True           # Input format: (batch_size, seq_len, features)
        )
        
        # Edge prediction MLP that processes vertex pair features
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concatenated vertex features input
            nn.LayerNorm(hidden_dim),                # Normalization for stable training
            nn.GELU(),                               # Smooth activation function
            nn.Dropout(0.1),                         # Regularization
            nn.Linear(hidden_dim, hidden_dim // 2),  # Compression layer
            nn.LayerNorm(hidden_dim // 2),           # Normalization
            nn.GELU(),                               # Activation
            nn.Linear(hidden_dim // 2, 1)            # Final edge probability prediction
        )
        
        # Cache for edge indices to avoid recomputation for same vertex count
        self.register_buffer('edge_indices_cache', None)
        self.cached_num_vertices = None
        
    def _get_edge_indices(self, num_vertices):
        """
        Generate all possible edge indices for a given number of vertices
        
        For N vertices, generates all unique pairs (i,j) where i < j.
        Results are cached to avoid recomputation for the same vertex count.
        
        Args:
            num_vertices (int): Number of vertices in the wireframe
            
        Returns:
            torch.Tensor: Edge indices of shape (num_edges, 2) where num_edges = N*(N-1)/2
        """
        # Check if we need to recompute edge indices
        if self.cached_num_vertices != num_vertices or self.edge_indices_cache is None:
            # Generate all unique vertex pairs (combinatorial approach)
            indices = []
            for i in range(num_vertices):
                for j in range(i + 1, num_vertices):  # Only upper triangle (i < j)
                    indices.append([i, j])
            
            # Cache the computed indices for efficiency
            self.edge_indices_cache = torch.tensor(indices, dtype=torch.long, device=next(self.parameters()).device)
            self.cached_num_vertices = num_vertices
            
        return self.edge_indices_cache
    
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
        
        # STEP 1: Project vertices to higher-dimensional feature space
        # Transform 3D coordinates to rich feature representation
        vertex_features = self.vertex_proj(vertices)  # (batch_size, num_vertices, hidden_dim)
        
        # STEP 2: Apply self-attention to capture vertex relationships
        # Self-attention allows each vertex to attend to all other vertices
        attended_features, _ = self.attention(
            vertex_features, vertex_features, vertex_features  # Query, Key, Value (all same for self-attention)
        )  # (batch_size, num_vertices, hidden_dim)
        
        # STEP 3: Add residual connection for better gradient flow
        vertex_features = vertex_features + attended_features
        
        # STEP 4: Generate edge indices for all possible vertex pairs
        edge_indices = self._get_edge_indices(num_vertices)  # (num_edges, 2)
        i_indices = edge_indices[:, 0]  # First vertex of each pair
        j_indices = edge_indices[:, 1]  # Second vertex of each pair
        
        # STEP 5: Gather features for each vertex pair
        # Extract features for vertices i and j in each potential edge
        v1 = vertex_features[:, i_indices, :]  # (batch_size, num_edges, hidden_dim) - first vertex features
        v2 = vertex_features[:, j_indices, :]  # (batch_size, num_edges, hidden_dim) - second vertex features
        
        # STEP 6: Predict edge probabilities
        # Concatenate vertex pair features and predict edge existence
        edge_features = torch.cat([v1, v2], dim=-1)  # (batch_size, num_edges, hidden_dim * 2)
        edge_features_flat = edge_features.view(-1, edge_features.shape[-1])  # Flatten for MLP processing
        
        # Apply edge prediction MLP
        edge_logits = self.edge_mlp(edge_features_flat)  # Raw predictions
        edge_probs = torch.sigmoid(edge_logits).view(batch_size, -1)  # Convert to probabilities [0,1]
        
        return edge_probs, edge_indices.tolist()  # Return probabilities and corresponding edge pairs
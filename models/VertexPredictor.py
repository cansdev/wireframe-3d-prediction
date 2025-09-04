
import torch
import torch.nn as nn

class VertexPredictor(nn.Module):
    """
    Dynamic vertex predictor for wireframe reconstruction
    
    This module predicts both the number of vertices and their 3D coordinates
    from global point cloud features. It handles variable vertex counts by:
    1. First predicting the vertex count using a classification head
    2. Then predicting vertex coordinates conditioned on the predicted count
    3. Using residual connections and deep MLPs for accurate coordinate regression
    
    The model can handle up to max_vertices and dynamically masks unused vertices.
    """
    
    def __init__(self, global_feature_dim=512, max_vertices=64, vertex_dim=3):
        """
        Initialize the dynamic vertex predictor
        
        Args:
            global_feature_dim (int): Dimension of input global features from encoder (default: 512)
            max_vertices (int): Maximum number of vertices the model can predict (default: 64)
            vertex_dim (int): Dimension of vertex coordinates (default: 3 for X,Y,Z)
        """
        super(VertexPredictor, self).__init__()
        
        self.max_vertices = max_vertices
        self.vertex_dim = vertex_dim
        
        # COMPONENT 1: Vertex count predictor - classifies how many vertices to predict
        # First predict the number of vertices with stronger regularization
        self.vertex_count_predictor = nn.Sequential(
            nn.Linear(global_feature_dim, 512),  # Expand features for count prediction
            nn.BatchNorm1d(512),                 # Batch normalization for stable training
            nn.ReLU(inplace=True),               # Non-linear activation
            nn.Dropout(0.5),                     # High dropout for regularization
            nn.Linear(512, 256),                 # Compression layer
            nn.BatchNorm1d(256),                 # Normalization
            nn.ReLU(inplace=True),               # Activation
            nn.Dropout(0.3),                     # Additional dropout
            nn.Linear(256, max_vertices + 1)     # Output: probability for each possible count (0 to max_vertices)
        )
        
        # COMPONENT 2: Deep MLPs for vertex coordinate prediction
        # Enhanced vertex coordinate predictor with increased capacity
        self.vertex_mlp1 = nn.Sequential(
            nn.Linear(global_feature_dim + 1, 4096),  # Input: global features + normalized count
            nn.LayerNorm(4096),                       # Layer normalization
            nn.ReLU(inplace=True),                    # Activation
            nn.Dropout(0.0),                          # No dropout for coordinate prediction
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
        
        # COMPONENT 3: Final output layer for all vertex coordinates
        # Dynamic output layer - predicts coordinates for maximum possible vertices
        self.final_layer = nn.Linear(1024, max_vertices * vertex_dim)  # Output: flattened coordinates
        
        # COMPONENT 4: Residual projection layers for better gradient flow
        # Residual connections with increased capacity
        self.residual_proj1 = nn.Linear(global_feature_dim + 1, 2048)  # First residual projection
        self.residual_proj2 = nn.Linear(global_feature_dim + 1, 1024)  # Second residual projection
        
    def forward(self, global_features, target_vertex_counts=None):
        """
        Forward pass for dynamic vertex prediction
        
        Args:
            global_features (torch.Tensor): Global features from point cloud encoder (batch_size, global_feature_dim)
            target_vertex_counts (torch.Tensor, optional): Ground truth vertex counts for training (batch_size,)
            
        Returns:
            dict: Dictionary containing:
                - vertices: Predicted vertex coordinates (batch_size, max_vertices, vertex_dim)
                - vertex_count_probs: Probability distribution over vertex counts (batch_size, max_vertices+1)
                - predicted_vertex_counts: Predicted number of vertices (batch_size,)
                - actual_vertex_counts: Actual counts used in computation (batch_size,)
        """
        batch_size = global_features.shape[0]
        
        # STEP 1: Predict vertex count using classification
        # Predict vertex counts with temperature scaling for sharper distributions
        vertex_count_logits = self.vertex_count_predictor(global_features)  # (batch_size, max_vertices+1)
        temperature = 0.1 if not self.training else 0.2  # Lower temperature for sharp predictions during inference
        vertex_count_probs = torch.softmax(vertex_count_logits / temperature, dim=-1)
        
        # STEP 2: Extract predicted vertex count from probability distribution
        # More aggressive argmax selection for discrete count prediction
        predicted_vertex_counts = torch.argmax(vertex_count_probs, dim=1)  # (batch_size,) - discrete count values
        
        # STEP 3: Determine actual vertex counts to use for coordinate prediction
        # Use target counts during training, predicted counts during inference
        if self.training and target_vertex_counts is not None:
            actual_vertex_counts = target_vertex_counts.float()  # Use ground truth for training
        else:
            actual_vertex_counts = predicted_vertex_counts.float()  # Use predictions for inference
        
        # STEP 4: Prepare input features for coordinate prediction
        # Normalize vertex counts to [0, 1] range for network input
        normalized_counts = actual_vertex_counts.unsqueeze(1) / self.max_vertices  # (batch_size, 1)
        
        # Concatenate global features with vertex count information
        enhanced_features = torch.cat([global_features, normalized_counts], dim=1)  # (batch_size, global_feature_dim+1)
        
        # STEP 5: Deep MLP processing for coordinate prediction with residual connections
        # Enhanced forward pass with vertex count conditioning
        x = self.vertex_mlp1(enhanced_features)  # First MLP layer
        x = self.vertex_mlp2(x)                  # Second MLP layer
        
        # First residual connection - helps with gradient flow
        residual1 = self.residual_proj1(enhanced_features)
        x = self.vertex_mlp3(x) + residual1      # Add residual and continue processing
        
        # Second residual connection - further improves gradient flow
        residual2 = self.residual_proj2(enhanced_features)
        x = self.vertex_mlp4(x) + residual2      # Add second residual
        
        # STEP 6: Generate final vertex coordinates
        # Final prediction - outputs all vertex coordinates simultaneously
        vertex_coords = self.final_layer(x)      # (batch_size, max_vertices * vertex_dim)
        vertex_coords = vertex_coords.view(batch_size, self.max_vertices, self.vertex_dim)  # Reshape to 3D coordinates
        
        # STEP 7: Mask unused vertices during inference
        # During inference, explicitly zero out vertices beyond predicted count
        if not self.training:
            for i in range(batch_size):
                count = predicted_vertex_counts[i].item()
                if count < self.max_vertices:
                    vertex_coords[i, count:, :] = 0.0  # Zero out unused vertices for clean output
        
        # Return comprehensive prediction results
        return {
            'vertices': vertex_coords,                            # Predicted 3D coordinates
            'vertex_count_probs': vertex_count_probs,            # Count probability distribution
            'predicted_vertex_counts': predicted_vertex_counts,   # Discrete predicted counts
            'actual_vertex_counts': actual_vertex_counts.long()   # Actual counts used in computation
        }
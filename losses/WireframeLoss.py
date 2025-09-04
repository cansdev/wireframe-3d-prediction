import torch
import torch.nn as nn


class WireframeLoss(nn.Module):
    """
    Combined multi-task loss function for wireframe prediction
    
    This loss combines four different objectives:
    1. Vertex Position Loss: MSE loss for 3D coordinate accuracy
    2. Edge Connectivity Loss: BCE loss for edge existence prediction
    3. Vertex Count Loss: Cross-entropy loss for count classification
    4. Sparsity Loss: L2 penalty to discourage over-prediction of vertices
    
    The loss is designed to handle variable vertex counts by masking
    and only computing losses on active vertices/edges.
    """
    
    def __init__(self, vertex_weight=1.0, edge_weight=1.0, count_weight=0.1, sparsity_weight=0.5):
        """
        Initialize the combined wireframe loss
        
        Args:
            vertex_weight (float): Weight for vertex position loss (default: 1.0)
            edge_weight (float): Weight for edge connectivity loss (default: 1.0) 
            count_weight (float): Weight for vertex count prediction loss (default: 0.1)
            sparsity_weight (float): Weight for sparsity regularization loss (default: 0.5)
        """
        super(WireframeLoss, self).__init__()
        
        # Store loss weights for different components
        self.vertex_weight = vertex_weight
        self.edge_weight = edge_weight
        self.count_weight = count_weight
        self.sparsity_weight = sparsity_weight
        
        # Initialize loss functions for different components
        self.mse_loss = nn.MSELoss()        # For vertex position regression
        self.bce_loss = nn.BCELoss()        # For edge existence classification
        self.ce_loss = nn.CrossEntropyLoss()  # For vertex count classification
        
    def forward(self, predictions, targets):
        """
        Compute the combined wireframe loss
        
        Args:
            predictions (dict): Model predictions containing:
                - vertices: Predicted vertex coordinates (batch_size, max_vertices, 3)
                - edge_probs: Edge existence probabilities (batch_size, num_edges)
                - vertex_count_probs: Vertex count probability distribution
                - predicted_vertex_counts: Discrete predicted vertex counts
                - actual_vertex_counts: Vertex counts used in computation
                
            targets (dict): Ground truth targets containing:
                - vertices: Target vertex coordinates (batch_size, max_vertices, 3)
                - edge_labels: Target edge existence labels (batch_size, num_edges)
                - vertex_counts: Target vertex counts (batch_size,)
                
        Returns:
            dict: Dictionary containing individual and total losses
        """
        batch_size = predictions['vertices'].shape[0]
        
        # COMPONENT 1: Vertex Position Loss (MSE on active vertices only)
        # Vertex position loss (only for active vertices)
        pred_vertices = predictions['vertices']      # (batch_size, max_vertices, 3)
        target_vertices = targets['vertices']        # (batch_size, max_vertices, 3)
        
        # Get actual vertex counts for proper masking
        if 'actual_vertex_counts' in predictions:
            vertex_counts = predictions['actual_vertex_counts']  # Use model's actual counts
        else:
            vertex_counts = targets['vertex_counts']             # Fallback to target counts
        
        # Calculate vertex loss only for active vertices (mask-aware computation)
        vertex_loss = 0
        for i in range(batch_size):
            count = vertex_counts[i].item()
            if count > 0:  # Only compute loss if there are vertices to predict
                pred_active = pred_vertices[i, :count, :]    # Active predicted vertices
                target_active = target_vertices[i, :count, :] # Active target vertices
                vertex_loss += self.mse_loss(pred_active, target_active)
        vertex_loss = vertex_loss / batch_size  # Average over batch
        
        # COMPONENT 2: Edge Connectivity Loss (BCE on predicted edge probabilities)
        # Edge connectivity loss
        pred_edge_probs = predictions['edge_probs']    # (batch_size, num_edges) - probabilities [0,1]
        target_edge_labels = targets['edge_labels']    # (batch_size, num_edges) - binary labels {0,1}
        
        # Handle variable edge counts by masking (edges depend on vertex count)
        if pred_edge_probs.numel() > 0 and target_edge_labels.numel() > 0:
            # Make sure dimensions match (handle cases where prediction and target have different edge counts)
            min_edges = min(pred_edge_probs.shape[1], target_edge_labels.shape[1])
            if min_edges > 0:
                pred_edges_masked = pred_edge_probs[:, :min_edges]      # Truncate to common size
                target_edges_masked = target_edge_labels[:, :min_edges]  # Truncate to common size
                edge_loss = self.bce_loss(pred_edges_masked, target_edges_masked)
            else:
                edge_loss = torch.tensor(0.0, device=pred_vertices.device)  # No edges case
        else:
            edge_loss = torch.tensor(0.0, device=pred_vertices.device)  # Empty predictions case
        
        # COMPONENT 3: Vertex Count Classification Loss (Cross-entropy on count predictions)
        # Vertex count prediction loss
        count_loss = torch.tensor(0.0, device=pred_vertices.device)
        if 'vertex_count_probs' in predictions and 'vertex_counts' in targets:
            pred_count_probs = predictions['vertex_count_probs']  # (batch_size, max_vertices+1) - softmax distribution
            target_counts = targets['vertex_counts']              # (batch_size,) - discrete count values
            # Ensure target counts are within valid range [0, max_vertices]
            target_counts = torch.clamp(target_counts, 0, pred_count_probs.shape[1] - 1)
            count_loss = self.ce_loss(pred_count_probs, target_counts)
        
        # COMPONENT 4: Sparsity Regularization Loss (L2 penalty on count over-prediction)
        # Add sparsity loss - penalize predicted vertex count more aggressively
        sparsity_loss = torch.tensor(0.0, device=pred_vertices.device)
        if 'predicted_vertex_counts' in predictions:
            pred_counts = predictions['predicted_vertex_counts'].float()  # Predicted discrete counts
            target_counts_float = targets['vertex_counts'].float()        # Target discrete counts
            # Use L2 loss for stronger penalty on large deviations from target count
            sparsity_loss = ((pred_counts - target_counts_float) ** 2).mean()
        
        # FINAL: Combine all loss components with respective weights
        # Combined loss with weighted sum of all components
        total_loss = (self.vertex_weight * vertex_loss +      # Coordinate accuracy
                     self.edge_weight * edge_loss +           # Connectivity accuracy  
                     self.count_weight * count_loss +         # Count classification
                     self.sparsity_weight * sparsity_loss)    # Over-prediction penalty
        
        # Return detailed loss breakdown for monitoring and debugging
        return {
            'total_loss': total_loss,        # Combined weighted loss
            'vertex_loss': vertex_loss,      # MSE loss for vertex positions
            'edge_loss': edge_loss,          # BCE loss for edge connectivity
            'count_loss': count_loss,        # CrossEntropy loss for count prediction
            'sparsity_loss': sparsity_loss   # L2 penalty for count regularization
        }

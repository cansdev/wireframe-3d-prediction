import torch
import torch.nn as nn


class WireframeLoss(nn.Module):
    """
    Combined multi-task loss function for wireframe prediction
    
    This loss combines two different objectives:
    1. Vertex Position Loss: MSE loss for 3D coordinate accuracy
    2. Edge Connectivity Loss: BCE loss for edge existence prediction
    
    The loss is designed to handle variable vertex counts by masking
    and only computing losses on active vertices/edges.
    """
    
    def __init__(self, vertex_weight=1.0, edge_weight=1.0, existence_weight=1.0):
        """
        Initialize the combined wireframe loss
        
        Args:
            vertex_weight (float): Weight for vertex position loss (default: 1.0)
            edge_weight (float): Weight for edge connectivity loss (default: 1.0)
            existence_weight (float): Weight for vertex existence loss (default: 1.0)
        """
        super(WireframeLoss, self).__init__()
        
        # Store loss weights for different components
        self.vertex_weight = vertex_weight
        self.edge_weight = edge_weight
        self.existence_weight = existence_weight
        
        # Initialize loss functions for different components
        self.mse_loss = nn.MSELoss()        # For vertex position regression
        self.bce_loss = nn.BCELoss()        # For edge existence classification
        
    def forward(self, predictions, targets):
        """
        Compute the combined wireframe loss
        
        Args:
            predictions (dict): Model predictions containing:
                - vertices: Predicted vertex coordinates (batch_size, max_vertices, 3)
                - existence_probabilities: Vertex existence probabilities (batch_size, max_vertices)
                - edge_probs: Edge existence probabilities (batch_size, num_edges)
                
            targets (dict): Ground truth targets containing:
                - vertices: Target vertex coordinates (batch_size, max_vertices, 3)
                - vertex_existence: Target vertex existence labels (batch_size, max_vertices)
                - edge_labels: Target edge existence labels (batch_size, num_edges)
                - vertex_counts: Target vertex counts (batch_size,)
                
        Returns:
            dict: Dictionary containing individual and total losses
        """
        batch_size = predictions['vertices'].shape[0]
        
        # COMPONENT 1: Vertex Position Loss (MSE on active vertices only)
        pred_vertices = predictions['vertices']      # (batch_size, max_vertices, 3)
        target_vertices = targets['vertices']        # (batch_size, max_vertices, 3)
        
        # Use target vertex existence for masking
        target_existence = targets['vertex_existence']  # (batch_size, max_vertices) - binary labels
        
        # Calculate vertex loss only for existing vertices using mask
        # Create mask for existing vertices
        mask = target_existence.unsqueeze(-1)  # (batch_size, max_vertices, 1)
        
        # Apply mask to both predictions and targets
        masked_pred_vertices = pred_vertices * mask
        masked_target_vertices = target_vertices * mask
        
        # Calculate MSE loss only on existing vertices
        vertex_loss = self.mse_loss(masked_pred_vertices, masked_target_vertices)
        
        # COMPONENT 2: Vertex Existence Loss (BCE on existence probabilities)
        pred_existence = predictions['existence_probabilities']  # (batch_size, max_vertices)
        existence_loss = self.bce_loss(pred_existence, target_existence.float())
        
        # COMPONENT 3: Edge Connectivity Loss (BCE on predicted edge probabilities)
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
        
        # FINAL: Combine all loss components with respective weights
        # Combined loss with weighted sum of all components
        total_loss = (self.vertex_weight * vertex_loss +      # Coordinate accuracy
                     self.existence_weight * existence_loss + # Vertex existence accuracy
                     self.edge_weight * edge_loss)            # Connectivity accuracy  
        
        # Return detailed loss breakdown for monitoring and debugging
        return {
            'total_loss': total_loss,          # Combined weighted loss
            'vertex_loss': vertex_loss,        # MSE loss for vertex positions
            'existence_loss': existence_loss,  # BCE loss for vertex existence
            'edge_loss': edge_loss,            # BCE loss for edge connectivity
        }

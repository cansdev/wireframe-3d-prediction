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
        self.smooth_l1_loss = nn.SmoothL1Loss()  # For vertex position regression (better than MSE)
        self.bce_loss = nn.BCELoss()             # For edge existence classification
        
    def forward(self, predictions, targets, matched_indices):
        """
        Compute the combined wireframe loss with Hungarian matching
        
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
                
            matched_indices (list): Hungarian matching indices for each batch element
                Each element is a tuple (pred_indices, target_indices) of matched pairs
                
        Returns:
            dict: Dictionary containing individual and total losses
        """
        batch_size = predictions['vertices'].shape[0]
        
        # COMPONENT 1: Vertex Position Loss (Smooth L1 on Hungarian matched vertices)
        pred_vertices = predictions['vertices']      # (batch_size, max_vertices, 3)
        target_vertices = targets['vertices']        # (batch_size, max_vertices, 3)
        target_existence = targets['vertex_existence']  # (batch_size, max_vertices) - binary labels
        
        # Use Hungarian matching for vertex loss computation
        vertex_loss = self._compute_matched_vertex_loss(pred_vertices, target_vertices, matched_indices)
        
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
            'vertex_loss': vertex_loss,        # Smooth L1 loss for vertex positions
            'existence_loss': existence_loss,  # BCE loss for vertex existence
            'edge_loss': edge_loss,            # BCE loss for edge connectivity
        }
    
    def _compute_matched_vertex_loss(self, pred_vertices, target_vertices, matched_indices):
        """
        Compute vertex loss using Hungarian matched pairs.
        
        Args:
            pred_vertices: Predicted vertices (batch_size, max_vertices, 3)
            target_vertices: Target vertices (batch_size, max_vertices, 3)
            matched_indices: List of (pred_indices, target_indices) tuples for each batch element
            
        Returns:
            torch.Tensor: Smooth L1 loss on matched vertex pairs
        """
        batch_size = pred_vertices.shape[0]
        device = pred_vertices.device
        
        total_loss = 0.0
        total_matches = 0
        
        for batch_idx in range(batch_size):
            pred_idx, target_idx = matched_indices[batch_idx]
            
            if len(pred_idx) > 0:
                # Get matched predictions and targets
                matched_pred = pred_vertices[batch_idx, pred_idx]  # (num_matches, 3)
                matched_target = target_vertices[batch_idx, target_idx]  # (num_matches, 3)
                
                # Compute Smooth L1 loss for this batch element
                batch_loss = self.smooth_l1_loss(matched_pred, matched_target)
                total_loss += batch_loss * len(pred_idx)  # Weight by number of matches
                total_matches += len(pred_idx)
        
        # Average over total number of matches across all batches
        if total_matches > 0:
            return total_loss / total_matches
        else:
            return torch.tensor(0.0, device=device)

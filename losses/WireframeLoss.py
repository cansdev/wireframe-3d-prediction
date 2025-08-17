import torch
import torch.nn as nn


class WireframeLoss(nn.Module):
    """Combined loss for vertex position, edge connectivity, and vertex count"""
    
    def __init__(self, vertex_weight=1.0, edge_weight=1.0, count_weight=0.1):
        super(WireframeLoss, self).__init__()
        self.vertex_weight = vertex_weight
        self.edge_weight = edge_weight
        self.count_weight = count_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions, targets):
        batch_size = predictions['vertices'].shape[0]
        
        # Vertex position loss (only for active vertices)
        pred_vertices = predictions['vertices']
        target_vertices = targets['vertices']
        
        # Get actual vertex counts for masking
        if 'actual_vertex_counts' in predictions:
            vertex_counts = predictions['actual_vertex_counts']
        else:
            vertex_counts = targets['vertex_counts']
        
        # Calculate vertex loss only for active vertices
        vertex_loss = 0
        for i in range(batch_size):
            count = vertex_counts[i].item()
            if count > 0:
                pred_active = pred_vertices[i, :count, :]
                target_active = target_vertices[i, :count, :]
                vertex_loss += self.mse_loss(pred_active, target_active)
        vertex_loss = vertex_loss / batch_size
        
        # Edge connectivity loss
        pred_edge_probs = predictions['edge_probs']
        target_edge_labels = targets['edge_labels']
        
        # Handle variable edge counts by masking
        if pred_edge_probs.numel() > 0 and target_edge_labels.numel() > 0:
            # Make sure dimensions match
            min_edges = min(pred_edge_probs.shape[1], target_edge_labels.shape[1])
            if min_edges > 0:
                pred_edges_masked = pred_edge_probs[:, :min_edges]
                target_edges_masked = target_edge_labels[:, :min_edges]
                edge_loss = self.bce_loss(pred_edges_masked, target_edges_masked)
            else:
                edge_loss = torch.tensor(0.0, device=pred_vertices.device)
        else:
            edge_loss = torch.tensor(0.0, device=pred_vertices.device)
        
        # Vertex count prediction loss
        count_loss = torch.tensor(0.0, device=pred_vertices.device)
        if 'vertex_count_probs' in predictions and 'vertex_counts' in targets:
            pred_count_probs = predictions['vertex_count_probs']
            target_counts = targets['vertex_counts'] - 1  # Adjust for 0-indexing
            target_counts = torch.clamp(target_counts, 0, pred_count_probs.shape[1] - 1)
            count_loss = self.ce_loss(pred_count_probs, target_counts)
        
        # Combined loss
        total_loss = (self.vertex_weight * vertex_loss + 
                     self.edge_weight * edge_loss + 
                     self.count_weight * count_loss)
        
        return {
            'total_loss': total_loss,
            'vertex_loss': vertex_loss,
            'edge_loss': edge_loss,
            'count_loss': count_loss
        }

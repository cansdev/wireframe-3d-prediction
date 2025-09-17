import torch
import numpy as np
import time
import logging
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

def create_adjacency_matrix_from_predictions(edge_probs, edge_indices, num_vertices, threshold=0.5):
    """Convert edge predictions to adjacency matrix"""
    batch_size = edge_probs.shape[0]
    adj_matrices = torch.zeros(batch_size, num_vertices, num_vertices)
    
    for batch_idx in range(batch_size):
        for edge_idx, (i, j) in enumerate(edge_indices):
            if edge_probs[batch_idx, edge_idx] > threshold:
                adj_matrices[batch_idx, i, j] = 1
                adj_matrices[batch_idx, j, i] = 1  # Symmetric
                
    return adj_matrices


def create_edge_labels_from_edge_set(edge_set, edge_indices):
    """Create edge labels tensor from edge set"""
    batch_size = 1  # Single example
    num_edges = len(edge_indices)
    edge_labels = torch.zeros(batch_size, num_edges)
    
    for edge_idx, (i, j) in enumerate(edge_indices):
        # Ensure consistent ordering (min, max)
        edge_tuple = (min(i, j), max(i, j))
        if edge_tuple in edge_set:
            edge_labels[0, edge_idx] = 1
            
    return edge_labels

def hungarian_rmse(pred_vertices, true_vertices):
    """Calculate RMSE using optimal Hungarian matching between predicted and true vertices"""
    if len(pred_vertices) == 0 and len(true_vertices) == 0:
        return 0.0  # Perfect match when both empty
    if len(pred_vertices) == 0 or len(true_vertices) == 0:
        return float('inf')  # Infinite error when one is empty
    
    # Create cost matrix (distances between all vertex pairs)
    costs = cdist(pred_vertices, true_vertices)
    
    # Find optimal matching using Hungarian algorithm
    pred_indices, true_indices = linear_sum_assignment(costs)
    
    # Calculate RMSE on optimally matched pairs
    matched_pred = pred_vertices[pred_indices]
    matched_true = true_vertices[true_indices]
    
    return np.sqrt(np.mean((matched_pred - matched_true) ** 2))

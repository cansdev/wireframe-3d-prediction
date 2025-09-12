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


def adaptive_lr_step(optimizer, loss_history, patience=100, factor=0.8, min_lr=1e-5):
    """
    Simple but super effective adaptive learning rate scheduler.
    Reduces LR by factor when loss doesn't improve for patience epochs.
    
    Args:
        optimizer: PyTorch optimizer
        loss_history: List of loss values
        patience: How many epochs to wait before reducing LR (increased to 100)
        factor: Factor to multiply LR by (0.8 = reduce by 20% instead of 50%)
        min_lr: Minimum learning rate threshold (increased to 1e-5)
        
    Returns:
        bool: True if LR was reduced, False otherwise
    """
    if len(loss_history) < patience + 1:
        return False
    
    # Check if loss hasn't improved in the last 'patience' epochs
    recent_losses = loss_history[-patience:]
    best_recent_loss = min(recent_losses)
    current_loss = loss_history[-1]
    
    # Only reduce if current loss is significantly worse (add small tolerance)
    improvement_threshold = 0.01  # Loss must improve by at least 1%
    if current_loss >= best_recent_loss * (1 + improvement_threshold):
        current_lr = optimizer.param_groups[0]['lr']
        
        # Only reduce if above minimum
        if current_lr > min_lr:
            new_lr = max(current_lr * factor, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            
            logger.info(f"Adaptive LR: Reduced from {current_lr:.8f} to {new_lr:.8f}")
            return True
    
    return False


class TrainingState:
    """Manages training state and progress tracking"""
    
    def __init__(self):
        self.best_loss = float('inf')
        self.best_vertex_rmse = float('inf')
        self.best_model_state = None
        self.loss_history = []
        self.vertex_rmse_history = []
        self.current_vertex_rmse = 999999
        self.start_time = time.time()
        
    def update_rmse(self, predictions, target_vertices, actual_vertex_counts, scalers):
        """Calculate and update current vertex RMSE"""
        with torch.no_grad():
            pred_vertices_np = predictions['vertices'].cpu().numpy()[0]
            target_vertices_np = target_vertices.cpu().numpy()[0]
            # Only use actual vertices for RMSE calculation
            actual_count = actual_vertex_counts[0].item()
            pred_vertices_actual = pred_vertices_np[:actual_count]
            target_vertices_actual = target_vertices_np[:actual_count]
            
            # Convert back to original scale for RMSE calculation using first scaler
            pred_vertices_orig = scalers[0].inverse_transform(pred_vertices_actual)
            target_vertices_orig = scalers[0].inverse_transform(target_vertices_actual)
            self.current_vertex_rmse = hungarian_rmse(pred_vertices_orig, target_vertices_orig)
            self.vertex_rmse_history.append(self.current_vertex_rmse)
            
    def update_best_state(self, model, total_loss):
        """Update best model state if current performance is better"""
        if self.current_vertex_rmse < self.best_vertex_rmse:
            self.best_vertex_rmse = self.current_vertex_rmse
            self.best_model_state = model.state_dict().copy()
        if total_loss.item() < self.best_loss:
            self.best_loss = total_loss.item()
            
    def track_loss(self, total_loss):
        """Track loss history"""
        self.loss_history.append(total_loss.item())
        
    def get_elapsed_time(self):
        """Get elapsed training time"""
        return time.time() - self.start_time
        
    def load_best_model(self, model):
        """Load the best model state"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model state with Vertex RMSE: {self.best_vertex_rmse:.6f}")
        return model


def log_training_progress(epoch, num_epochs, total_loss, loss_dict, current_vertex_rmse, 
                         current_lr, elapsed_time):
    """Log training progress with detailed metrics"""
    logger.info(f"Epoch {epoch:4d}/{num_epochs} | "
               f"Total: {total_loss.item():.6f} | "
               f"Vertex: {loss_dict['vertex_loss'].item():.6f} | "
               f"Existence: {loss_dict['existence_loss'].item():.6f} | "
               f"Edge: {loss_dict['edge_loss'].item():.6f}")
    
    logger.info(f"           RMSE: {current_vertex_rmse:.6f} | "
               f"LR: {current_lr:.8f} | "
               f"Time: {elapsed_time:.1f}s")


def create_wandb_log_dict(epoch, total_loss, loss_dict, current_vertex_rmse, 
                         current_lr, elapsed_time, best_loss, best_vertex_rmse):
    """Create dictionary for wandb logging"""
    return {
        "epoch": epoch,
        "total_loss": total_loss.item(),
        "vertex_loss": loss_dict['vertex_loss'].item(),
        "existence_loss": loss_dict['existence_loss'].item(),
        "edge_loss": loss_dict['edge_loss'].item(),
        "vertex_rmse": current_vertex_rmse,
        "learning_rate": current_lr,
        "elapsed_time": elapsed_time,
        "best_loss": best_loss,
        "best_vertex_rmse": best_vertex_rmse
    }

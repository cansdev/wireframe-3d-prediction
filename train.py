import time
import numpy as np
from scipy.optimize import linear_sum_assignment # hungarian algorithm
from scipy.spatial.distance import cdist # distance matrix
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
import logging
from models.PointCloudToWireframe import PointCloudToWireframe
from losses.WireframeLoss import WireframeLoss
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def train_overfit_model(batch_data, num_epochs=5000, learning_rate=0.001, wandb_run=None):
    """Train model to overfit on batch of examples"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Extract data from batch_data dictionary
    point_clouds = batch_data['point_clouds']  # Shape: (batch_size, num_points, 3)
    vertices = batch_data['vertices']  # Shape: (batch_size, max_vertices, 3)
    original_samples = batch_data['original_samples']
    
    # Get dimensions
    batch_size, _, _ = point_clouds.shape
    _, max_vertices, _ = vertices.shape
    
    # Extract actual vertex counts for each sample
    actual_vertex_counts = []
    for sample in original_samples:
        actual_vertex_counts.append(len(sample.vertices))
    actual_vertex_counts = torch.tensor(actual_vertex_counts, dtype=torch.long).to(device)
    
    # Create model with increased capacity for batch training
    model = PointCloudToWireframe(input_dim=8, max_vertices=max_vertices).to(device)
    
    # Print model parameters for verification
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: Total={total_params:,}, Trainable={trainable_params:,}")
    
    # Prepare data tensors
    point_cloud_tensor = torch.FloatTensor(point_clouds).to(device)
    target_vertices = torch.FloatTensor(vertices).to(device)
    
    # Create edge labels for each sample in the batch using edge sets
    edge_labels_list = []
    for i in range(batch_size):
        actual_count = actual_vertex_counts[i].item()
        sample_edge_set = original_samples[i].edge_set
        edge_labels = create_edge_labels_from_edge_set(
            sample_edge_set, 
            [(j, k) for j in range(actual_count) for k in range(j+1, actual_count)]
        )
        edge_labels_list.append(edge_labels.squeeze(0))  # Remove the batch dimension
    
    # Pad edge labels to same length for batch processing
    max_edges = max([len(labels) for labels in edge_labels_list]) if edge_labels_list else 0
    if max_edges > 0:
        edge_labels_batch = torch.zeros(batch_size, max_edges).to(device)
        for i, labels in enumerate(edge_labels_list):
            if len(labels) > 0:
                edge_labels_batch[i, :len(labels)] = labels
    else:
        edge_labels_batch = torch.zeros(batch_size, 0).to(device)
    
    # Create vertex existence labels for each sample in the batch
    vertex_existence_batch = torch.zeros(batch_size, max_vertices).to(device)
    for i in range(batch_size):
        actual_count = actual_vertex_counts[i].item()
        vertex_existence_batch[i, :actual_count] = 1.0  # Mark existing vertices as 1
    
    criterion = WireframeLoss(
        vertex_weight=3.0,  # Reduced vertex weight 
        edge_weight=1.0,    # Increased edge weight significantly
        existence_weight=1.5  # Weight for vertex existence prediction
    ) 
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6, eps=1e-8, betas=(0.9, 0.999))  # Add small weight decay
    
    # 5. Fine-tuning parameters (define first)
    warmup_epochs = 30  # Reduced warmup
    warmup_factor = 0.01
    fine_tuning_start = 200  # Start fine-tuning when performance is stable
    # REMOVED: ultra_fine_tuning_start - causes RMSE degradation
    
    # Enhanced adaptive learning rate system with better convergence
    # 1. Cosine Annealing with reasonable minimum learning rate
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=learning_rate * 0.1  # Much higher min LR
    )
    
    # 2. Plateau reduction with conservative parameters
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=50, threshold=0.005, 
        threshold_mode='rel', cooldown=20, min_lr=learning_rate * 1e-3  # Higher min LR
    )

    # 3. Multi-component learning rates for different loss components
    vertex_lr_multiplier = 1.0
    edge_lr_multiplier = 1.2
    existence_lr_multiplier = 1.0
    
    # Training loop
    model.train()
    best_loss = float('inf')
    best_vertex_rmse = float('inf')  # Initialize best vertex RMSE
    best_model_state = None  # Initialize best model state
    loss_history = []
    vertex_rmse_history = []
    patience = 500  # Further reduced patience for faster stopping when not improving
    patience_counter = 0
    current_vertex_rmse = 999999

    logger.info("=" * 80)
    logger.info(f"Starting batch training for {num_epochs} epochs...")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max vertices: {max_vertices}")
    logger.info(f"Target vertex counts: {actual_vertex_counts.cpu().numpy()}")
    logger.info(f"Loss weights - Vertex: {criterion.vertex_weight}, Edge: {criterion.edge_weight}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info("=" * 80)
    start_time = time.time()

    # W&B run is now managed by main.py
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass with vertex count information
        predictions = model(point_cloud_tensor, actual_vertex_counts)
        
        # Calculate loss
        targets = {
            'vertices': target_vertices,
            'vertex_existence': vertex_existence_batch,
            'edge_labels': edge_labels_batch,
            'vertex_counts': actual_vertex_counts
        }
        loss_dict = criterion(predictions, targets)
        total_loss = loss_dict['total_loss']
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Enhanced learning rate scheduling with phase-based optimization
        current_lr_base = optimizer.param_groups[0]['lr']
        
        # 1. Warmup phase
        if epoch < warmup_epochs:
            warmup_lr = learning_rate * (warmup_factor + (1.0 - warmup_factor) * epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        elif epoch < fine_tuning_start:
            # Phase 1: Normal cosine annealing for initial convergence
            scheduler_cosine.step()  # Removed deprecated epoch parameter
        else:
            # Phase 2: Fine-tuning phase - much more conservative decay
            if epoch == fine_tuning_start:
                logger.info(f"Entering fine-tuning phase at epoch {epoch}")
            # Use very conservative decay to maintain learning capability
            decay_factor = 0.99995  # Very slow decay
            current_lr = current_lr_base * (decay_factor ** (epoch - fine_tuning_start))
            min_lr = learning_rate * 0.01  # Much higher minimum LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(current_lr, min_lr)
        
        # Dynamic loss weight adjustment based on training progress
        if epoch > 0 and epoch % 20 == 0:
            # Calculate loss component ratios
            vertex_loss_ratio = loss_dict['vertex_loss'].item() / total_loss.item()
            existence_loss_ratio = loss_dict['existence_loss'].item() / total_loss.item()
            edge_loss_ratio = loss_dict['edge_loss'].item() / total_loss.item()
            
            # Dynamic loss weight adjustment for fine convergence
            # REMOVED: vertex_weight_multiplier logic - no sparsity handling needed
            
            # Learning rate multiplier adjustments (more conservative)
            if vertex_loss_ratio > 0.1:  # Lowered thresholds
                vertex_lr_multiplier *= 1.1  # Increase vertex LR when it's significant
            elif vertex_loss_ratio < 0.001:
                vertex_lr_multiplier *= 0.95  # Slight reduction if vertex loss is tiny
                
            # Edge loss adjustment
            if edge_loss_ratio > 0.05:
                edge_lr_multiplier *= 1.05
            elif edge_loss_ratio < 0.001:
                edge_lr_multiplier *= 0.98
                
            # Existence loss adjustment
            if existence_loss_ratio > 0.1:
                existence_lr_multiplier *= 1.1
            elif existence_loss_ratio < 0.001:
                existence_lr_multiplier *= 0.95
            
            # Clamp multipliers with wider ranges for vertex precision
            vertex_lr_multiplier = max(0.1, min(5.0, vertex_lr_multiplier))  # Allow higher vertex LR
            edge_lr_multiplier = max(0.1, min(3.0, edge_lr_multiplier))
            existence_lr_multiplier = max(0.1, min(4.0, existence_lr_multiplier))  # Balance between vertex and edge
            
            # REMOVED: vertex_weight_multiplier clamping - no longer using sparsity weights
        
        # Track progress
        loss_history.append(total_loss.item())
        
        # Calculate current vertex RMSE for monitoring (using first sample in batch)
        with torch.no_grad():
            pred_vertices_np = predictions['vertices'].cpu().numpy()[0]
            target_vertices_np = target_vertices.cpu().numpy()[0]
            # Only use actual vertices for RMSE calculation
            actual_count = actual_vertex_counts[0].item()
            pred_vertices_actual = pred_vertices_np[:actual_count]
            target_vertices_actual = target_vertices_np[:actual_count]
            
            # Convert back to original scale for RMSE calculation using first scaler
            scalers = batch_data['scalers']
            pred_vertices_orig = scalers[0].inverse_transform(pred_vertices_actual)
            target_vertices_orig = scalers[0].inverse_transform(target_vertices_actual)
            current_vertex_rmse = hungarian_rmse(pred_vertices_orig, target_vertices_orig)
            vertex_rmse_history.append(current_vertex_rmse)
        
        # Apply plateau scheduler based on vertex RMSE (more conservative)
        if epoch >= warmup_epochs and epoch % 5 == 0:  # Less frequent application
            scheduler_plateau.step(current_vertex_rmse)
        
        # Additional fine-tuning plateau detection with higher thresholds
        if epoch >= fine_tuning_start and current_vertex_rmse < 1.0:  # Higher threshold
            # Apply very moderate LR reduction only when really needed
            if epoch % 30 == 0 and current_vertex_rmse > 0.1:  # Less frequent and higher threshold
                current_lr = optimizer.param_groups[0]['lr']
                if current_lr > learning_rate * 1e-3:  # Only reduce if above minimum
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.95  # Very gentle reduction
        
        # Early stopping based on vertex RMSE and save best model
        if current_vertex_rmse < best_vertex_rmse:
            best_vertex_rmse = current_vertex_rmse
            best_model_state = model.state_dict().copy()  # Save the best model state
            patience_counter = 0
        else:
            patience_counter += 1
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
        
        # Early stopping check (improved for better convergence)
        if patience_counter >= patience and epoch > 250:  # Allow more time for initial convergence
            logger.info(f"Early stopping at epoch {epoch}! Vertex RMSE hasn't improved for {patience} epochs")
            break

        # Log progress with detailed metrics every 20 epochs
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            elapsed_time = time.time() - start_time
            
            # Get current learning rate from optimizer
            current_lr = optimizer.param_groups[0]['lr']
            
            # Calculate additional metrics for logging
            with torch.no_grad():
                # Get learning rate from optimizer
                current_lr = optimizer.param_groups[0]['lr']
            
            logger.info(f"Epoch {epoch:4d}/{num_epochs} | "
                       f"Total: {total_loss.item():.6f} | "
                       f"Vertex: {loss_dict['vertex_loss'].item():.6f} | "
                       f"Existence: {loss_dict['existence_loss'].item():.6f} | "
                       f"Edge: {loss_dict['edge_loss'].item():.6f}")
            
            logger.info(f"           RMSE: {current_vertex_rmse:.6f} | "
                       f"LR: {current_lr:.8f} | "
                       f"Time: {elapsed_time:.1f}s")
            
            # Enhanced learning rate logging
            if epoch > warmup_epochs and epoch % 40 == 0:
                logger.info(f"           LR Multipliers - Vertex: {vertex_lr_multiplier:.3f}, "
                           f"Edge: {edge_lr_multiplier:.3f}")
                
                # Log loss component ratios
                vertex_ratio = loss_dict['vertex_loss'].item() / total_loss.item()
                existence_ratio = loss_dict['existence_loss'].item() / total_loss.item()
                edge_ratio = loss_dict['edge_loss'].item() / total_loss.item()
                
                logger.info(f"           Loss Ratios - Vertex: {vertex_ratio:.3f}, "
                           f"Existence: {existence_ratio:.3f}, "
                           f"Edge: {edge_ratio:.3f}")

            # Log comprehensive metrics to wandb
            if wandb_run is not None:
                log_dict = {
                    "epoch": epoch,
                    "total_loss": total_loss.item(),
                    "vertex_loss": loss_dict['vertex_loss'].item(),
                    "existence_loss": loss_dict['existence_loss'].item(),
                    "edge_loss": loss_dict['edge_loss'].item(),
                    "vertex_rmse": current_vertex_rmse,
                    "learning_rate": current_lr,
                    "elapsed_time": elapsed_time,
                    "best_loss": best_loss,
                    "best_vertex_rmse": best_vertex_rmse,
                    "patience_counter": patience_counter
                }
                
                wandb_run.log(log_dict)

            # Log separator for readability
            if epoch % 100 == 0 or epoch == num_epochs - 1:
                logger.info("-" * 80)  # Add separator line for readability
            
    # Load the best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model state with Vertex RMSE: {best_vertex_rmse:.6f}")

    logger.info(f"Training completed! Best loss: {best_loss:.6f}")
    
    return model, loss_history


def evaluate_model(model, batch_data, device, max_vertices):
    """Evaluate the trained model on batch data"""
    model.eval()
    
    # Extract data from batch_data dictionary
    point_clouds = batch_data['point_clouds']
    vertices = batch_data['vertices']
    scalers = batch_data['scalers']
    original_samples = batch_data['original_samples']
    
    results = []
    
    with torch.no_grad():
        # Prepare input - convert entire batch
        point_cloud_tensor = torch.FloatTensor(point_clouds).to(device)
        
        # Forward pass without ground truth hints (pure inference)
        predictions = model(point_cloud_tensor, target_vertex_counts=None)
        
        # Process each sample in the batch
        for i in range(len(point_clouds)):
            # Get predictions for this sample
            pred_vertices_full = predictions['vertices'][i].cpu().numpy()
            pred_edge_probs = predictions['edge_probs'][i].cpu().numpy()
            edge_indices = predictions['edge_indices'][i]
            
            # Get original sample for this sample
            original_sample = original_samples[i]
            scaler = scalers[i]
            
            # Use predicted vertex count from existence probabilities
            predicted_vertex_count = predictions['actual_vertex_counts'][i].item()

            # Use only the predicted number of vertices for evaluation
            pred_vertices = pred_vertices_full[:predicted_vertex_count]
            
            # Convert back to original scale - only the actual vertices
            if predicted_vertex_count > 0:
                pred_vertices_original = scaler.inverse_transform(pred_vertices)
            else:
                pred_vertices_original = np.array([]).reshape(0, 3)
            
            true_vertices_original = original_sample.vertices
            
            # Problem 1: Calculate RMSE using Hungarian matching
            # Calculates RMSE using only the first sample in the batch, could be using all samples and taking mean
            vertex_rmse = hungarian_rmse(pred_vertices_original, true_vertices_original)
            
            # Edge accuracy (threshold at 0.5)
            # Generate edge indices for the actual number of vertices
            actual_edge_indices = edge_indices
            
            # Only take the edge probabilities for valid edges (based on actual vertices)
            num_actual_edges = len(actual_edge_indices)
            pred_edge_probs_actual = pred_edge_probs[:num_actual_edges]
            
            pred_adj_matrix = create_adjacency_matrix_from_predictions(
                torch.FloatTensor(pred_edge_probs_actual).unsqueeze(0),
                actual_edge_indices,
                predicted_vertex_count,
                threshold=0.3
            )[0].numpy()
            
            # Problem 2: Adjacency matrix for true edges using predicted vertex count for compatibility
            # True adjacency matrix is (16,16) while predicted adjacency matrix is (6,6)
            true_edge_set = original_sample.edge_set
            true_vertex_count = len(true_vertices_original)
            true_adj_matrix = np.zeros((predicted_vertex_count, predicted_vertex_count))
            for edge_tuple in true_edge_set:
                v1, v2 = edge_tuple
                if v1 < predicted_vertex_count and v2 < predicted_vertex_count:
                    true_adj_matrix[v1, v2] = 1
                    true_adj_matrix[v2, v1] = 1
            
            edge_accuracy = np.mean((pred_adj_matrix == true_adj_matrix).astype(float))
            
            # Edge precision and recall
            true_edges = (true_adj_matrix == 1)
            pred_edges = (pred_adj_matrix == 1)
            
            tp = np.sum(true_edges & pred_edges)
            fp = np.sum(~true_edges & pred_edges)
            fn = np.sum(true_edges & ~pred_edges)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            result = {
                'sample_index': i,
                'vertex_rmse': vertex_rmse,
                'edge_accuracy': edge_accuracy,
                'edge_precision': precision,
                'edge_recall': recall,
                'edge_f1_score': f1_score,
                'predicted_vertices': pred_vertices_original,
                'true_vertices': true_vertices_original,
                'predicted_adjacency': pred_adj_matrix,
                'edge_probabilities': pred_edge_probs,
                'predicted_vertex_count': predicted_vertex_count,
                'true_vertex_count': len(true_vertices_original)
            }
            
            results.append(result)
    
    return results

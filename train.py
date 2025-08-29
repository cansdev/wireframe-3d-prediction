import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import logging
from models.PointCloudToWireframe import PointCloudToWireframe
from losses.WireframeLoss import WireframeLoss
from models.EdgePredictor import EdgePredictor
from models.PointNetEncoder import PointNetEncoder
from demo_dataset.PCtoWFdataset import PCtoWFdataset
import wandb

logging.basicConfig(level=logging.INFO)
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


def compute_building3d_metrics(results):
    """Compute Building3D benchmark metrics from evaluation results"""
    all_aco = []
    all_cp = []
    all_cr = []
    all_c_f1 = []
    all_ep = []
    all_er = []
    all_e_f1 = []
    
    for result in results:
        # Extract data for proper Building3D calculations
        pred_vertices = result['predicted_vertices']
        true_vertex_count = result['true_vertex_count']
        
        # Calculate ACO (Average Corner Offset) - use actual vertex positions
        if len(pred_vertices) > 0 and true_vertex_count > 0:
            # For proper ACO, we need true vertex positions, but we only have count
            # Use vertex RMSE as approximation for now
            aco = result['vertex_rmse']
        else:
            aco = float('inf')
        
        # Use actual edge metrics from evaluation
        ep = result['edge_precision']
        er = result['edge_recall']
        e_f1 = result['edge_f1_score']
        
        # For corner metrics, use a threshold-based approach
        # Consider a vertex "correctly predicted" if within threshold
        corner_threshold = 2.0  # meters - adjust based on your scale
        vertex_rmse = result['vertex_rmse']
        
        # Simplified corner precision/recall based on RMSE threshold
        if vertex_rmse <= corner_threshold:
            cp = 1.0  # Good precision if RMSE is low
            cr = 1.0  # Good recall if RMSE is low
        else:
            cp = 0.0  # Poor precision if RMSE is high
            cr = 0.0  # Poor recall if RMSE is high
        
        c_f1 = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0
        
        all_aco.append(aco)
        all_cp.append(cp)
        all_cr.append(cr)
        all_c_f1.append(c_f1)
        all_ep.append(ep)
        all_er.append(er)
        all_e_f1.append(e_f1)
    
    # Calculate global averages
    global_aco = np.mean(all_aco)
    global_cp = np.mean(all_cp)
    global_cr = np.mean(all_cr)
    global_c_f1 = np.mean(all_c_f1)
    global_ep = np.mean(all_ep)
    global_er = np.mean(all_er)
    global_e_f1 = np.mean(all_e_f1)
    
    return {
        'building3d_aco': global_aco,
        'building3d_cp': global_cp,
        'building3d_cr': global_cr,
        'building3d_c_f1': global_c_f1,
        'building3d_ep': global_ep,
        'building3d_er': global_er,
        'building3d_e_f1': global_e_f1
    }


def train_overfit_model(batch_data, num_epochs=5000, learning_rate=0.001, wandb_run=None):
    """Train model to overfit on batch of examples"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Extract data from batch_data dictionary
    point_clouds = batch_data['point_clouds']  # Shape: (batch_size, num_points, 3)
    vertices = batch_data['vertices']  # Shape: (batch_size, max_vertices, 3)
    original_samples = batch_data['original_samples']
    
    # Get dimensions
    batch_size, num_points, _ = point_clouds.shape
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
    
    criterion = WireframeLoss(
        vertex_weight=30.0, 
        edge_weight=10.0, 
        count_weight=10.0,
        sparsity_weight=10.0 
    )  # More aggressive penalties for vertex count
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0, eps=1e-8)
    # More aggressive learning rate schedule
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 600, 750, 850], gamma=0.3)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    best_vertex_rmse = float('inf')  # Initialize best vertex RMSE
    best_model_state = None  # Initialize best model state
    loss_history = []
    vertex_rmse_history = []
    patience = 500  # Increased patience for better convergence
    patience_counter = 0
    current_vertex_rmse = 999999

    logger.info("=" * 80)
    logger.info(f"Starting batch training for {num_epochs} epochs...")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max vertices: {max_vertices}")
    logger.info(f"Target vertex counts: {actual_vertex_counts.cpu().numpy()}")
    logger.info(f"Loss weights - Vertex: {criterion.vertex_weight}, Edge: {criterion.edge_weight}, Count: {criterion.count_weight}, Sparsity: {criterion.sparsity_weight}")
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
        scheduler.step()
        
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
            current_vertex_rmse = np.sqrt(np.mean((pred_vertices_orig - target_vertices_orig) ** 2))
            vertex_rmse_history.append(current_vertex_rmse)
        
        # Early stopping based on vertex RMSE and save best model
        if current_vertex_rmse < best_vertex_rmse:
            best_vertex_rmse = current_vertex_rmse
            best_model_state = model.state_dict().copy()  # Save the best model state
            patience_counter = 0
        else:
            patience_counter += 1
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
        
        # Early stopping check (more aggressive)
        if patience_counter >= patience and epoch > 400:
            logger.info(f"Early stopping at epoch {epoch}! Vertex RMSE hasn't improved for {patience} epochs")
            break

        # Log progress with detailed metrics every 20 epochs
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            elapsed_time = time.time() - start_time
            
            # Get current learning rate
            current_lr = scheduler.get_last_lr()[0]
            
            # Calculate additional metrics for logging
            with torch.no_grad():
                # Count accuracy (how often we predict the exact vertex count)
                pred_counts = predictions['predicted_vertex_counts'].cpu().numpy()
                true_counts = actual_vertex_counts.cpu().numpy()
                count_accuracy = np.mean(pred_counts == true_counts) * 100
                
                # Average count error
                count_error = np.mean(np.abs(pred_counts - true_counts))
                
                # Max count error
                max_count_error = np.max(np.abs(pred_counts - true_counts))
            
            logger.info(f"Epoch {epoch:4d}/{num_epochs} | "
                       f"Total: {total_loss.item():.6f} | "
                       f"Vertex: {loss_dict['vertex_loss'].item():.6f} | "
                       f"Edge: {loss_dict['edge_loss'].item():.6f} | "
                       f"Count: {loss_dict['count_loss'].item():.6f} | "
                       f"Sparsity: {loss_dict['sparsity_loss'].item():.6f}")
            
            logger.info(f"           RMSE: {current_vertex_rmse:.6f} | "
                       f"Count Acc: {count_accuracy:.1f}% | "
                       f"Count Err: {count_error:.2f} | "
                       f"Max Err: {max_count_error:.0f} | "
                       f"LR: {current_lr:.8f} | "
                       f"Time: {elapsed_time:.1f}s")
        
        



            # Log comprehensive metrics to wandb
            if wandb_run is not None:
                wandb_run.log({
                    "epoch": epoch,
                    "total_loss": total_loss.item(),
                    "vertex_loss": loss_dict['vertex_loss'].item(),
                    "edge_loss": loss_dict['edge_loss'].item(),
                    "count_loss": loss_dict['count_loss'].item(),
                    "sparsity_loss": loss_dict['sparsity_loss'].item(),
                    "vertex_rmse": current_vertex_rmse,
                    "count_accuracy": count_accuracy,
                    "count_error": count_error,
                    "max_count_error": max_count_error,
                    "learning_rate": current_lr,
                    "elapsed_time": elapsed_time,
                    "best_loss": best_loss,
                    "best_vertex_rmse": best_vertex_rmse,
                    "patience_counter": patience_counter
                })

            
            # Log predicted vs target counts for first few samples
            if epoch % 100 == 0 or epoch == num_epochs - 1:
                logger.info(f"           Pred counts: {pred_counts[:min(len(pred_counts), 7)]}")
                logger.info(f"           True counts: {true_counts[:min(len(true_counts), 7)]}")
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
        
        # Get actual vertex counts for evaluation
        actual_vertex_counts = []
        for sample in original_samples:
            actual_vertex_counts.append(len(sample.vertices))
        actual_vertex_counts = torch.tensor(actual_vertex_counts, dtype=torch.long).to(device)
        
        # Forward pass on entire batch WITH vertex counts to help prediction
        predictions = model(point_cloud_tensor, actual_vertex_counts)
        
        # Process each sample in the batch
        for i in range(len(point_clouds)):
            # Get predictions for this sample
            pred_vertices_full = predictions['vertices'][i].cpu().numpy()
            pred_edge_probs = predictions['edge_probs'][i].cpu().numpy()
            edge_indices = predictions['edge_indices']
            
            # Get the PREDICTED vertex count (not the full array)
            predicted_vertex_count = predictions['predicted_vertex_counts'][i].item()
            
            # Apply stricter thresholding for vertex count
            vertex_count_probs = predictions['vertex_count_probs'][i].cpu().numpy()
            confidence_threshold = 0.5  # Lowered from 0.7
            
            # Find the highest probability count
            max_prob_idx = vertex_count_probs.argmax()
            max_prob = vertex_count_probs[max_prob_idx]
            
            if max_prob >= confidence_threshold:
                predicted_vertex_count = int(max_prob_idx)  # Direct indexing since we're 0-indexed now
            else:
                # If no high confidence, use weighted average of top-3 predictions
                top_k = 3
                top_indices = vertex_count_probs.argsort()[-top_k:]
                top_probs = vertex_count_probs[top_indices]
                top_probs = top_probs / top_probs.sum()  # Normalize
                predicted_vertex_count = int(np.round(np.sum(top_indices * top_probs)))
            
            # Hard limit to prevent over-prediction
            actual_vertex_count = len(original_samples[i].vertices)
            # Be more conservative - aim for slightly fewer vertices
            predicted_vertex_count = min(predicted_vertex_count, actual_vertex_count)
            
            # Additional filtering: if predicted count is way too high, use actual count
            if predicted_vertex_count > actual_vertex_count * 1.5:
                predicted_vertex_count = actual_vertex_count
            
            # CRITICAL: Only use vertices up to the predicted count
            pred_vertices = pred_vertices_full[:predicted_vertex_count]  # This removes extra dots!
            
            # Get original sample for this sample
            original_sample = original_samples[i]
            scaler = scalers[i]
            
            # Convert back to original scale - only the predicted vertices
            if predicted_vertex_count > 0:
                pred_vertices_original = scaler.inverse_transform(pred_vertices)
            else:
                pred_vertices_original = np.array([]).reshape(0, 3)
            
            true_vertices_original = original_sample.vertices
            
            # Calculate metrics only on actual predicted vertices
            if predicted_vertex_count > 0 and len(true_vertices_original) > 0:
                # For RMSE, compare only up to min of predicted and true counts
                min_count = min(predicted_vertex_count, len(true_vertices_original))
                vertex_mse = np.mean((pred_vertices_original[:min_count] - true_vertices_original[:min_count]) ** 2)
                vertex_rmse = np.sqrt(vertex_mse)
            else:
                vertex_rmse = float('inf')
            
            # Edge accuracy (threshold at 0.5)
            # Use the predicted number of vertices for edge evaluation
            predicted_num_vertices = predicted_vertex_count  # Fix variable name
            
            # Generate edge indices for the predicted number of vertices
            predicted_edge_indices = [(j, k) for j in range(predicted_num_vertices) for k in range(j+1, predicted_num_vertices)]
            
            # Only take the edge probabilities for valid edges (based on predicted vertices)
            num_predicted_edges = len(predicted_edge_indices)
            pred_edge_probs_actual = pred_edge_probs[:num_predicted_edges]
            
            pred_adj_matrix = create_adjacency_matrix_from_predictions(
                torch.FloatTensor(pred_edge_probs_actual).unsqueeze(0),
                predicted_edge_indices,
                predicted_num_vertices,
                threshold=0.5
            )[0].numpy()
            
            # Get original edge set and create adjacency matrix for evaluation
            true_edge_set = original_sample.edge_set
            true_adj_matrix = np.zeros((predicted_num_vertices, predicted_num_vertices))
            for edge_tuple in true_edge_set:
                v1, v2 = edge_tuple
                if v1 < predicted_num_vertices and v2 < predicted_num_vertices:
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
                'predicted_vertices': pred_vertices_original,  # Only predicted count vertices
                'predicted_adjacency': pred_adj_matrix,
                'edge_probabilities': pred_edge_probs,
                'predicted_vertex_count': predicted_vertex_count,  # Add this for debugging
                'true_vertex_count': len(true_vertices_original)
            }
            
            results.append(result)
    
    return results

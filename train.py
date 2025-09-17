import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import logging
from models.PointCloudToWireframe import PointCloudToWireframe
from models.utils import (
    create_edge_labels_from_edge_set, 
)
from losses.WireframeLoss import WireframeLoss
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(data_loader, num_epochs=5000, learning_rate=0.001, wandb_run=None):

    """Train model using the new Building3D dataset structure"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Get first batch to determine model dimensions
    first_batch = next(iter(data_loader))
    
    # Extract data from the new dataset format
    point_clouds = first_batch['point_clouds']  # Shape: (batch_size, num_points, features)
    wf_vertices = first_batch['wf_vertices']  # List of variable length vertex arrays
    wf_edges = first_batch['wf_edges']  # List of variable length edge arrays
    
    # Get dimensions
    batch_size = point_clouds.shape[0]
    num_points, input_dim = point_clouds.shape[1], point_clouds.shape[2]
    
    # Find max vertices across all samples in the dataset
    max_vertices = max(len(vertices) for vertices in wf_vertices)
    
    # Create model with appropriate dimensions
    model = PointCloudToWireframe(input_dim=input_dim, max_vertices=max_vertices).to(device)
    
    # Print model parameters for verification
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: Total={total_params:,}, Trainable={trainable_params:,}")
    
    # Prepare data tensors
    point_cloud_tensor = point_clouds.to(device)
    
    # Create vertex existence labels for each sample in the batch
    vertex_existence_batch = torch.zeros(batch_size, max_vertices).to(device)
    actual_vertex_counts = []
    
    for i in range(batch_size):
        actual_count = len(wf_vertices[i])
        actual_vertex_counts.append(actual_count)
        vertex_existence_batch[i, :actual_count] = 1.0  # Mark existing vertices as 1
    
    actual_vertex_counts = torch.tensor(actual_vertex_counts, dtype=torch.long).to(device)
    
    # Create edge labels for each sample in the batch
    edge_labels_list = []
    for i in range(batch_size):
        actual_count = actual_vertex_counts[i].item()
        sample_edges = wf_edges[i]
        
        # Create edge set from the edges
        edge_set = set()
        for edge in sample_edges:
            v1, v2 = edge[0].item(), edge[1].item()
            edge_set.add((min(v1, v2), max(v1, v2)))
        
        # Create all possible edges for this sample
        all_possible_edges = [(j, k) for j in range(actual_count) for k in range(j+1, actual_count)]
        
        # Create edge labels
        edge_labels = create_edge_labels_from_edge_set(edge_set, all_possible_edges)
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
        vertex_weight=3.0,  # Reduced vertex weight 
        edge_weight=1.0,    # Increased edge weight significantly
        existence_weight=1.5  # Weight for vertex existence prediction
    ) 
    
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6, eps=1e-8, betas=(0.9, 0.999))  # Add small weight decay
    

    
    # Training loop
    model.train()

    logger.info("=" * 80)
    logger.info(f"Starting batch training for {num_epochs} epochs...")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max vertices: {max_vertices}")
    logger.info(f"Target vertex counts: {actual_vertex_counts.cpu().numpy()}")
    logger.info(f"Loss weights - Vertex: {criterion.vertex_weight}, Edge: {criterion.edge_weight}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info("=" * 80)
    start_time = time.time()

    # Create target vertices tensor for loss calculation
    target_vertices = torch.zeros(batch_size, max_vertices, 3).to(device)
    for i in range(batch_size):
        actual_count = actual_vertex_counts[i].item()
        target_vertices[i, :actual_count] = wf_vertices[i][:actual_count]
    
    # Simple training state variables (no class needed)
    best_loss = float('inf')
    best_vertex_rmse = float('inf')
    best_model_state = None
    loss_history = []
    patience_counter = 0
    patience = 100  # Early stopping patience
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(point_cloud_tensor, actual_vertex_counts)
        
        # Calculate loss (Hungarian matching handled internally)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track loss
        loss_history.append(total_loss.item())
        
        # Calculate RMSE for monitoring (simple version)
        with torch.no_grad():
            pred_vertices = predictions['vertices'][0].cpu().numpy()[:actual_vertex_counts[0].item()]
            true_vertices = target_vertices[0].cpu().numpy()[:actual_vertex_counts[0].item()]
            current_vertex_rmse = np.sqrt(np.mean((pred_vertices - true_vertices) ** 2))
        
        # Update best model
        if current_vertex_rmse < best_vertex_rmse:
            best_vertex_rmse = current_vertex_rmse
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}! No improvement for {patience} epochs")
            break
        
        # Log progress every 20 epochs
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            elapsed_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            # Simple logging
            logger.info(f"Epoch {epoch:4d}/{num_epochs} | "
                       f"Loss: {total_loss.item():.6f} | "
                       f"RMSE: {current_vertex_rmse:.6f} | "
                       f"LR: {current_lr:.8f} | "
                       f"Time: {elapsed_time:.1f}s")
            
            # Log to wandb if available
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
                    "best_vertex_rmse": best_vertex_rmse
                }
                wandb_run.log(log_dict)
            
            if epoch % 100 == 0:
                logger.info("-" * 80)
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with RMSE: {best_vertex_rmse:.6f}")
    
    logger.info(f"Training completed! Best loss: {best_loss:.6f}")
    return model

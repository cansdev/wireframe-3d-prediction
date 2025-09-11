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
from models.LearningScheduler import (
    AdaptiveLearningScheduler, TrainingState, log_training_progress, 
    create_wandb_log_dict, hungarian_rmse, create_adjacency_matrix_from_predictions,
    create_edge_labels_from_edge_set
)
from losses.WireframeLoss import WireframeLoss
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    
    # Initialize learning scheduler and training state
    learning_scheduler = AdaptiveLearningScheduler(optimizer, num_epochs, learning_rate)
    training_state = TrainingState()
    
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
        
        # Track progress
        training_state.track_loss(total_loss)
        
        # Calculate current vertex RMSE for monitoring (using first sample in batch)
        training_state.update_rmse(predictions, target_vertices, actual_vertex_counts, batch_data['scalers'])
        
        # Save best model state for monitoring
        training_state.update_best_state(model, total_loss)

        # Log progress with detailed metrics every 20 epochs
        if epoch % 20 == 0 or epoch == num_epochs - 1:
            elapsed_time = training_state.get_elapsed_time()
            
            # Get current learning rate from optimizer
            current_lr = optimizer.param_groups[0]['lr']
            
            # Use the new logging function
            log_training_progress(epoch, num_epochs, total_loss, loss_dict, 
                                training_state.current_vertex_rmse, current_lr, 
                                elapsed_time)

            # Log comprehensive metrics to wandb
            if wandb_run is not None:
                log_dict = create_wandb_log_dict(epoch, total_loss, loss_dict, 
                                               training_state.current_vertex_rmse, current_lr, 
                                               elapsed_time, training_state.best_loss, 
                                               training_state.best_vertex_rmse)
                wandb_run.log(log_dict)

            # Log separator for readability
            if epoch % 100 == 0 or epoch == num_epochs - 1:
                logger.info("-" * 80)  # Add separator line for readability
            
    # Load the best model state before returning
    model = training_state.load_best_model(model)

    logger.info(f"Training completed! Best loss: {training_state.best_loss:.6f}")
    
    return model, training_state.loss_history




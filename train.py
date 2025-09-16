import time
import numpy as np
from scipy.optimize import linear_sum_assignment # hungarian algorithm
from scipy.spatial.distance import cdist # distance matrix
import torch
import torch.nn as nn
from torch.optim import Adam
import logging
from models.PointCloudToWireframe import PointCloudToWireframe
from models.WireframeHungarianMatcher import WireframeHungarianMatcher
from models.LearningScheduler import (
    TrainingState, log_training_progress, 
    create_wandb_log_dict, hungarian_rmse, create_adjacency_matrix_from_predictions,
    create_edge_labels_from_edge_set, adaptive_lr_step
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
    wf_edge_number = first_batch['wf_edge_number']  # Number of edges per sample
    
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
    
    # Initialize Hungarian matcher for optimal vertex assignment
    hungarian_matcher = WireframeHungarianMatcher(
        cost_vertex=1.0,      # Weight for vertex coordinate matching
        cost_existence=1.0    # Weight for existence probability matching
    )
    
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6, eps=1e-8, betas=(0.9, 0.999))  # Add small weight decay
    
    # Initialize training state
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

    # Create target vertices tensor for loss calculation
    target_vertices = torch.zeros(batch_size, max_vertices, 3).to(device)
    for i in range(batch_size):
        actual_count = actual_vertex_counts[i].item()
        target_vertices[i, :actual_count] = wf_vertices[i][:actual_count]
    
    # W&B run is now managed by main.py
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass with vertex count information
        predictions = model(point_cloud_tensor, actual_vertex_counts)
        
        # Prepare targets for Hungarian matching
        targets_for_matching = []
        for i in range(batch_size):
            actual_count = actual_vertex_counts[i].item()
            target_dict = {
                'vertices': target_vertices[i, :actual_count],  # Only existing vertices
                'existence': torch.ones(actual_count, device=device)  # All existing vertices have existence=1
            }
            targets_for_matching.append(target_dict)
        
        # Perform Hungarian matching
        with torch.no_grad():
            matched_indices = hungarian_matcher(predictions, targets_for_matching)
        
        # Calculate loss with Hungarian matching
        targets = {
            'vertices': target_vertices,
            'vertex_existence': vertex_existence_batch,
            'edge_labels': edge_labels_batch,
            'vertex_counts': actual_vertex_counts
        }
        loss_dict = criterion(predictions, targets, matched_indices)
        total_loss = loss_dict['total_loss']
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track progress
        training_state.track_loss(total_loss)
        
        # Calculate current vertex RMSE for monitoring (using first sample in batch)
        # For now, we'll use a simple RMSE calculation without scalers
        pred_vertices = predictions['vertices']
        vertex_rmse = torch.sqrt(torch.mean((pred_vertices - target_vertices) ** 2))
        training_state.current_vertex_rmse = vertex_rmse.item()
        
        # Save best model state for monitoring
        training_state.update_best_state(model, total_loss)
        
        # Apply adaptive learning rate (every 50 epochs to avoid too frequent changes)
        if epoch > 0 and epoch % 50 == 0:
            adaptive_lr_step(optimizer, training_state.loss_history, patience=150, factor=0.85)

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

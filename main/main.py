import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import logging
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PointCloudWireframeDataset:
    """Dataset class for loading and preprocessing point cloud and wireframe data"""
    
    def __init__(self, xyz_file, obj_file):
        self.xyz_file = xyz_file
        self.obj_file = obj_file
        self.point_cloud = None
        self.vertices = None
        self.edges = None
        self.edge_adjacency_matrix = None
        
    def load_point_cloud(self):
        """Load point cloud data from XYZ file"""
        logger.info(f"Loading point cloud from {self.xyz_file}")
        data = []
        
        with open(self.xyz_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:  # X Y Z R G B A Intensity
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    r, g, b, a = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                    intensity = float(parts[7])
                    data.append([x, y, z, r, g, b, a, intensity])
        
        self.point_cloud = np.array(data)
        logger.info(f"Loaded {len(self.point_cloud)} points")
        return self.point_cloud
    
    def load_wireframe(self):
        """Load wireframe data from OBJ file"""
        logger.info(f"Loading wireframe from {self.obj_file}")
        vertices = []
        edges = []
        
        with open(self.obj_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 0:
                    continue
                    
                if parts[0] == 'v':  # Vertex
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append([x, y, z])
                elif parts[0] == 'l':  # Line (edge)
                    # OBJ format uses 1-based indexing
                    v1, v2 = int(parts[1]) - 1, int(parts[2]) - 1
                    edges.append([v1, v2])
        
        self.vertices = np.array(vertices)
        self.edges = np.array(edges)
        
        logger.info(f"Loaded {len(self.vertices)} vertices and {len(self.edges)} edges")
        return self.vertices, self.edges
    
    def create_adjacency_matrix(self):
        """Create adjacency matrix from edges"""
        if self.vertices is None or self.edges is None:
            raise ValueError("Must load wireframe data first")
            
        n_vertices = len(self.vertices)
        self.edge_adjacency_matrix = np.zeros((n_vertices, n_vertices))
        
        for edge in self.edges:
            v1, v2 = edge[0], edge[1]
            self.edge_adjacency_matrix[v1, v2] = 1
            self.edge_adjacency_matrix[v2, v1] = 1  # Undirected graph
            
        return self.edge_adjacency_matrix
    
    def normalize_data(self):
        """Normalize point cloud and vertex coordinates"""
        if self.point_cloud is None:
            raise ValueError("Must load point cloud first")
            
        # Normalize spatial coordinates (X, Y, Z)
        spatial_coords = self.point_cloud[:, :3]
        self.spatial_scaler = StandardScaler()
        normalized_spatial = self.spatial_scaler.fit_transform(spatial_coords)
        
        # Normalize color values (R, G, B, A) to [0, 1]
        color_vals = self.point_cloud[:, 3:7] / 255.0
        
        # Normalize intensity
        intensity = self.point_cloud[:, 7:8]
        self.intensity_scaler = StandardScaler()
        normalized_intensity = self.intensity_scaler.fit_transform(intensity)
        
        # Combine normalized features
        self.normalized_point_cloud = np.hstack([
            normalized_spatial, color_vals, normalized_intensity
        ])
        
        # Normalize vertex coordinates using same spatial scaler
        if self.vertices is not None:
            self.normalized_vertices = self.spatial_scaler.transform(self.vertices)
        
        logger.info("Data normalization completed")
        return self.normalized_point_cloud
    
    def find_nearest_points_to_vertices(self, k=5):
        """Find k nearest points for each vertex in the wireframe"""
        if self.normalized_point_cloud is None or self.normalized_vertices is None:
            raise ValueError("Must normalize data first")
            
        # Use only spatial coordinates for nearest neighbor search
        point_spatial = self.normalized_point_cloud[:, :3]
        vertex_spatial = self.normalized_vertices[:, :3]
        
        # Find k nearest points for each vertex
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(point_spatial)
        distances, indices = nbrs.kneighbors(vertex_spatial)
        
        self.vertex_to_points_mapping = {
            'distances': distances,
            'indices': indices
        }
        
        logger.info(f"Found {k} nearest points for each of {len(self.normalized_vertices)} vertices")
        return distances, indices


class PointNetEncoder(nn.Module):
    """PointNet-inspired encoder for point cloud features"""
    
    def __init__(self, input_dim=8, hidden_dims=[64, 128, 256], output_dim=512):
        super(PointNetEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x):
        # x shape: (batch_size, num_points, input_dim)
        batch_size, num_points, input_dim = x.shape
        
        # Reshape for MLP processing
        x = x.view(-1, input_dim)  # (batch_size * num_points, input_dim)
        
        # Apply MLP to each point
        point_features = self.mlp(x)  # (batch_size * num_points, output_dim)
        
        # Reshape back
        point_features = point_features.view(batch_size, num_points, -1)
        
        # Global max pooling across points
        # Transpose for pooling: (batch_size, output_dim, num_points)
        point_features = point_features.transpose(1, 2)
        global_features = self.global_pool(point_features).squeeze(-1)
        
        return global_features, point_features.transpose(1, 2)


class VertexPredictor(nn.Module):
    """Predict vertex locations from global point cloud features"""
    
    def __init__(self, global_feature_dim=512, num_vertices=32, vertex_dim=3):
        super(VertexPredictor, self).__init__()
        
        self.num_vertices = num_vertices
        self.vertex_dim = vertex_dim
        
        self.vertex_mlp = nn.Sequential(
            nn.Linear(global_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_vertices * vertex_dim)
        )
        
    def forward(self, global_features):
        # Predict vertex coordinates
        vertex_coords = self.vertex_mlp(global_features)
        vertex_coords = vertex_coords.view(-1, self.num_vertices, self.vertex_dim)
        return vertex_coords


class EdgePredictor(nn.Module):
    """Predict edge connectivity between vertices"""
    
    def __init__(self, vertex_dim=3, hidden_dim=128):
        super(EdgePredictor, self).__init__()
        
        # Edge features are concatenated vertex features
        self.edge_mlp = nn.Sequential(
            nn.Linear(vertex_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, vertices):
        # vertices shape: (batch_size, num_vertices, vertex_dim)
        batch_size, num_vertices, vertex_dim = vertices.shape
        
        # Create all possible pairs of vertices
        edges = []
        edge_indices = []
        
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):  # Only upper triangle to avoid duplicate edges
                # Concatenate vertex features
                v1 = vertices[:, i, :]  # (batch_size, vertex_dim)
                v2 = vertices[:, j, :]  # (batch_size, vertex_dim)
                edge_feature = torch.cat([v1, v2], dim=1)  # (batch_size, vertex_dim * 2)
                edges.append(edge_feature)
                edge_indices.append((i, j))
        
        # Stack all edge features
        edge_features = torch.stack(edges, dim=1)  # (batch_size, num_edges, vertex_dim * 2)
        
        # Reshape for MLP processing
        batch_size, num_edges = edge_features.shape[:2]
        edge_features = edge_features.view(-1, vertex_dim * 2)
        
        # Predict edge probabilities
        edge_probs = self.edge_mlp(edge_features)  # (batch_size * num_edges, 1)
        edge_probs = edge_probs.view(batch_size, num_edges)
        
        return edge_probs, edge_indices


class PointCloudToWireframe(nn.Module):
    """Complete model for point cloud to wireframe prediction"""
    
    def __init__(self, input_dim=8, num_vertices=32):
        super(PointCloudToWireframe, self).__init__()
        
        self.num_vertices = num_vertices
        
        # Point cloud encoder
        self.encoder = PointNetEncoder(input_dim=input_dim)
        
        # Vertex predictor
        self.vertex_predictor = VertexPredictor(
            global_feature_dim=512, 
            num_vertices=num_vertices
        )
        
        # Edge predictor
        self.edge_predictor = EdgePredictor(vertex_dim=3)
        
    def forward(self, point_cloud):
        # Encode point cloud
        global_features, point_features = self.encoder(point_cloud)
        
        # Predict vertices
        predicted_vertices = self.vertex_predictor(global_features)
        
        # Predict edges
        edge_probs, edge_indices = self.edge_predictor(predicted_vertices)
        
        return {
            'vertices': predicted_vertices,
            'edge_probs': edge_probs,
            'edge_indices': edge_indices,
            'global_features': global_features
        }


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


def create_edge_labels_from_adjacency(adj_matrix, edge_indices):
    """Create edge labels tensor from adjacency matrix"""
    batch_size = 1  # Single example
    num_edges = len(edge_indices)
    edge_labels = torch.zeros(batch_size, num_edges)
    
    for edge_idx, (i, j) in enumerate(edge_indices):
        if adj_matrix[i, j] == 1:
            edge_labels[0, edge_idx] = 1
            
    return edge_labels


class WireframeLoss(nn.Module):
    """Combined loss for vertex position and edge connectivity"""
    
    def __init__(self, vertex_weight=1.0, edge_weight=1.0):
        super(WireframeLoss, self).__init__()
        self.vertex_weight = vertex_weight
        self.edge_weight = edge_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, predictions, targets):
        # Vertex position loss
        pred_vertices = predictions['vertices']
        target_vertices = targets['vertices']
        vertex_loss = self.mse_loss(pred_vertices, target_vertices)
        
        # Edge connectivity loss
        pred_edge_probs = predictions['edge_probs']
        target_edge_labels = targets['edge_labels']
        edge_loss = self.bce_loss(pred_edge_probs, target_edge_labels)
        
        # Combined loss
        total_loss = self.vertex_weight * vertex_loss + self.edge_weight * edge_loss
        
        return {
            'total_loss': total_loss,
            'vertex_loss': vertex_loss,
            'edge_loss': edge_loss
        }


def train_overfit_model(dataset, num_epochs=5000, learning_rate=0.001):
    """Train model to overfit on single example"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")
    
    # Create model
    num_vertices = len(dataset.vertices)
    model = PointCloudToWireframe(input_dim=8, num_vertices=num_vertices).to(device)
    
    # Prepare data
    point_cloud_tensor = torch.FloatTensor(dataset.normalized_point_cloud).unsqueeze(0).to(device)
    target_vertices = torch.FloatTensor(dataset.normalized_vertices).unsqueeze(0).to(device)
    
    # Create edge labels
    edge_labels = create_edge_labels_from_adjacency(
        dataset.edge_adjacency_matrix, 
        [(i, j) for i in range(num_vertices) for j in range(i+1, num_vertices)]
    ).to(device)
    
    # Loss function and optimizer
    criterion = WireframeLoss(vertex_weight=1.0, edge_weight=2.0)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.8)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    loss_history = []
    
    logger.info(f"Starting overtraining for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(point_cloud_tensor)
        
        # Calculate loss
        targets = {
            'vertices': target_vertices,
            'edge_labels': edge_labels
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
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            
        # Log progress
        if epoch % 100 == 0 or epoch == num_epochs - 1:
            elapsed_time = time.time() - start_time
            logger.info(f"Epoch {epoch:4d}/{num_epochs} | "
                       f"Total Loss: {total_loss.item():.6f} | "
                       f"Vertex Loss: {loss_dict['vertex_loss'].item():.6f} | "
                       f"Edge Loss: {loss_dict['edge_loss'].item():.6f} | "
                       f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                       f"Time: {elapsed_time:.1f}s")
    
    logger.info(f"Training completed! Best loss: {best_loss:.6f}")
    
    return model, loss_history


def evaluate_model(model, dataset, device):
    """Evaluate the trained model"""
    model.eval()
    
    with torch.no_grad():
        # Prepare input
        point_cloud_tensor = torch.FloatTensor(dataset.normalized_point_cloud).unsqueeze(0).to(device)
        
        # Forward pass
        predictions = model(point_cloud_tensor)
        
        # Get predictions
        pred_vertices = predictions['vertices'].cpu().numpy()[0]
        pred_edge_probs = predictions['edge_probs'].cpu().numpy()[0]
        edge_indices = predictions['edge_indices']
        
        # Convert back to original scale
        pred_vertices_original = dataset.spatial_scaler.inverse_transform(pred_vertices)
        true_vertices_original = dataset.vertices
        
        # Calculate metrics
        vertex_mse = np.mean((pred_vertices_original - true_vertices_original) ** 2)
        vertex_rmse = np.sqrt(vertex_mse)
        
        # Edge accuracy (threshold at 0.5)
        pred_adj_matrix = create_adjacency_matrix_from_predictions(
            torch.FloatTensor(pred_edge_probs).unsqueeze(0),
            edge_indices,
            len(dataset.vertices),
            threshold=0.5
        )[0].numpy()
        
        true_adj_matrix = dataset.edge_adjacency_matrix
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
        
        results = {
            'vertex_rmse': vertex_rmse,
            'edge_accuracy': edge_accuracy,
            'edge_precision': precision,
            'edge_recall': recall,
            'edge_f1_score': f1_score,
            'predicted_vertices': pred_vertices_original,
            'predicted_adjacency': pred_adj_matrix,
            'edge_probabilities': pred_edge_probs
        }
        
        return results


# Load and preprocess the data
def load_and_preprocess_data():
    """Load and preprocess the main dataset"""
    dataset = PointCloudWireframeDataset('1.xyz', '1.obj')
    
    # Load data
    point_cloud = dataset.load_point_cloud()
    vertices, edges = dataset.load_wireframe()
    
    # Create adjacency matrix
    adj_matrix = dataset.create_adjacency_matrix()
    
    # Normalize data
    normalized_pc = dataset.normalize_data()
    
    # Find nearest points to vertices
    distances, indices = dataset.find_nearest_points_to_vertices(k=10)
    
    return dataset

if __name__ == "__main__":
    # Load data
    dataset = load_and_preprocess_data()
    
    print(f"Point cloud shape: {dataset.point_cloud.shape}")
    print(f"Vertices shape: {dataset.vertices.shape}") 
    print(f"Edges shape: {dataset.edges.shape}")
    print(f"Adjacency matrix shape: {dataset.edge_adjacency_matrix.shape}")
    print(f"Normalized point cloud shape: {dataset.normalized_point_cloud.shape}")
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*50)
    print("STARTING OVERTRAINING ON SINGLE EXAMPLE")
    print("="*50)
    
    model, loss_history = train_overfit_model(dataset, num_epochs=2000, learning_rate=0.001)
    
    print("\n" + "="*50)
    print("EVALUATING TRAINED MODEL")
    print("="*50)
    
    # Evaluate model
    results = evaluate_model(model, dataset, device)
    
    print(f"Vertex RMSE: {results['vertex_rmse']:.6f}")
    print(f"Edge Accuracy: {results['edge_accuracy']:.4f}")
    print(f"Edge Precision: {results['edge_precision']:.4f}")
    print(f"Edge Recall: {results['edge_recall']:.4f}")
    print(f"Edge F1-Score: {results['edge_f1_score']:.4f}")
    
    # Save model
    torch.save(model.state_dict(), 'trained_model.pth')
    print("\nModel saved as 'trained_model.pth'")
    
    print("\nOvertraining completed successfully!")

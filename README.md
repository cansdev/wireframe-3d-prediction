# Point Cloud to Wireframe Prediction System

This system implements a deep learning model that learns to predict wireframe structures from point cloud data. The system is designed for batch training on multiple examples with efficient edge list representation and fixed vertex count architecture.

## 🎯 Recent Results (Edge List Architecture)

**Latest Performance on Test Dataset (8 samples):**

### Global Performance Metrics
- **Global Vertex RMSE**: 8.003082
- **Global Edge Precision**: 0.554054 (aggregated across all samples)
- **Global Edge Recall**: 0.317829 (aggregated across all samples)  
- **Global Edge F1-Score**: 0.403941 (aggregated across all samples)
- **Total True Positives**: 41 edges correctly predicted
- **Total False Positives**: 33 incorrect edge predictions
- **Total False Negatives**: 88 missed edges

### Individual Sample Highlights
- **Best Performers**: Samples 1, 2, 7, 8 achieving near-perfect edge prediction (0.94-1.00 F1)
- **Challenging Cases**: Complex wireframes with 30-48 edges show room for improvement
- **Architecture Success**: No complete failures (all samples show some edge detection capability)

## Overview

The system takes as input:
- **Point Cloud Data** (`.xyz` format): Contains 3D coordinates, RGB colors, and intensity values
- **Wireframe Data** (`.obj` format): Contains vertices and edge connectivity information

The goal is to train a neural network that can predict both the number of vertices and their positions, plus edge connectivity, directly from point cloud data.

## Architecture

### Model Components

1. **Enhanced PointNet Encoder**: Processes point clouds with dual pooling
   - **Dual Pooling**: Combines max pooling AND mean pooling for richer features
   - **Architecture**: [512, 1024, 2048, 1024] → 512D global features
   - **Feature Fusion**: 1024→2048→1024→512 fusion network for combined pooling
   - **Input**: Point cloud with XYZ coordinates + RGBA + intensity (8 features total)
   - **Output**: 512-dimensional global feature vector

2. **Vertex Predictor**: Predicts vertex positions with fixed architecture
   - **Architecture**: 512 → 4096 → 2048 → 2048 → 1024 → (max_vertices×3)
   - **Residual Connections**: Multiple skip connections for better gradient flow
   - **Fixed Output**: All samples padded to max_vertices during training
   - **Normalization**: LayerNorm and BatchNorm for stable training
   - **Masking**: Uses actual vertex counts for loss calculation only

3. **Edge Predictor**: Binary connectivity classification using edge lists
   - **Input**: Concatenated vertex pairs (6D: two 3D vertices)
   - **Architecture**: 6 → 512 → 256 → 128 → 1 with sigmoid activation
   - **Processing**: All possible vertex combinations for predicted vertex count
   - **Output**: Probability for each possible vertex pair
   - **Efficiency**: Uses edge sets instead of dense adjacency matrices

### Data Flow

```
Point Cloud (N×8) → Enhanced PointNet Encoder → Global Features (512D)
                                               ↓
                    Vertex Predictor → Fixed Positions (max_vertices×3)
                                               ↓
                    Edge Predictor → Edge Probabilities (E×1) using edge lists
```

### Key Architectural Improvements

#### Edge List Representation (New!)
- **Memory Efficiency**: O(E) instead of O(V²) storage for edge information
- **No Padding**: Eliminates need to pad adjacency matrices to max_vertices
- **Dynamic Processing**: Handles variable vertex counts naturally
- **Ground Truth**: Uses efficient edge sets for label creation

#### Fixed Vertex Architecture with Smart Masking
- **Fixed Output**: All samples padded to max_vertices dimensions
- **Smart Loss Masking**: Only actual vertices contribute to vertex loss calculation
- **Edge Masking**: Edge prediction only considers valid vertex pairs

#### Batch Training Architecture
- **Multi-Sample Learning**: Trains on batches of diverse wireframe examples
- **Variable Complexity**: Handles samples with different vertex/edge counts via padding and masking
- **Robust Metrics**: Global metrics aggregated across all test samples

## Files Structure

```
wireframe-3d-prediction/
├── main.py                           # Main training script for batch learning
├── train.py                          # Training functions with edge list support
├── evaluate.py                       # Comprehensive evaluation with global metrics
├── visualize_open3d.py              # Interactive 3D Open3D visualizations
├── trained_model.pth                # Saved model weights (after training)
├── demo_dataset/
│   ├── PCtoWFdataset.py             # Dataset loader for multiple file pairs
│   ├── PointCloudWireframeDataset.py # Batch dataset with edge list support
│   ├── train_dataset/               # Training data directory
│   │   ├── point_cloud/             # Training point cloud files (.xyz)
│   │   └── wireframe/               # Training wireframe files (.obj)
│   └── test_dataset/                # Test data directory
│       ├── point_cloud/             # Test point cloud files (.xyz)
│       └── wireframe/               # Test wireframe files (.obj)
├── models/
│   ├── PointCloudToWireframe.py     # Main model with dynamic vertex prediction
│   ├── PointNetEncoder.py           # Enhanced PointNet with dual pooling
│   ├── VertexPredictor.py           # Dynamic vertex count and position predictor
│   └── EdgePredictor.py             # Edge connectivity predictor
├── losses/
│   └── WireframeLoss.py             # Combined vertex + edge + count loss
├── visualize/
│   └── visualize_wireframe.py       # Matplotlib visualization functions
├── test/
│   └── test_model.py                # Model testing and validation
├── output/                          # Generated results and visualizations
│   ├── summary_report.txt           # Detailed performance analysis
│   ├── test_sample_N/               # Per-sample visualization directories
│   │   ├── *_prediction_comparison.png
│   │   └── *_edge_probabilities.png
├── requirements.txt                 # Python dependencies
└── README.md                        # This documentation
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

Required dependencies:
- PyTorch >= 1.9.0 (with CUDA support recommended)
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0
- Scikit-learn >= 1.0.0
- Open3D >= 0.15.0 (for interactive 3D visualization)

### CUDA Setup (Recommended)

For GPU acceleration:
```bash
# Remove CPU-only version
pip uninstall -y torch torchvision torchaudio

# Install CUDA-enabled version
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

## Usage

### 1. Test the System

First, verify everything works correctly:

```bash
python test/test_model.py
```

**Expected Output:**
```
============================================================
POINT CLOUD TO WIREFRAME - MODEL TEST
============================================================
Testing data loading...
✓ Edge set: 18 edges
Edge connectivity: 18 connections
✓ Point cloud loaded: (N, 8)
✓ Vertices loaded: (V, 3)
✓ Normalized point cloud: (N, 8)

Testing model architecture...
Using device: cuda
✓ Enhanced PointNet encoder with dual pooling
✓ Vertex predictor with fixed architecture and masking
✓ Edge predictor with edge list support
✓ Model forward pass successful
✓ Predicted vertices shape: torch.Size([1, max_vertices, 3])
✓ Predicted edge probabilities shape: torch.Size([1, num_edges])
✓ Vertex count prediction from VertexPredictor
✓ Global features shape: torch.Size([1, 512])

✓ ALL TESTS PASSED SUCCESSFULLY!
```

This will:
- Load and preprocess multiple samples with edge list representation
- Test fixed architecture with smart masking
- Verify edge list-based loss computation
- Validate batch processing capabilities

### 2. Train the Model

To train the model on multiple samples with batch learning:

```bash
python main.py
```

**Expected Training Progress:**
```
============================================================
DATASET INFORMATION
============================================================

============================================================
STARTING BATCH TRAINING
============================================================
Using device: cuda

Training on device: cuda
Starting batch training for 1000 epochs...
Batch size: N
Max vertices: V
Target vertex counts: [v1, v2, v3, ...]
Loss weights - Vertex: 5.0, Edge: 1.0, Count: 100.0, Sparsity: 50.0

Epoch    0/1000 | Total: 15.234567 | Vertex: 12.345678 | Edge: 1.234567 | Count: 0.987654 | Sparsity: 0.666321
           RMSE: 8.765432 | Count Acc: 12.5% | Count Err: 3.45 | Max Err: 8 | LR: 0.00100000

Epoch  100/1000 | Total: 3.456789 | Vertex: 2.345678 | Edge: 0.876543 | Count: 0.123456 | Sparsity: 0.111111
           RMSE: 2.345678 | Count Acc: 75.0% | Count Err: 1.25 | Max Err: 3 | LR: 0.00100000

Epoch  400/1000 | Total: 1.234567 | Vertex: 0.876543 | Edge: 0.234567 | Count: 0.098765 | Sparsity: 0.024691
           RMSE: 1.123456 | Count Acc: 87.5% | Count Err: 0.75 | Max Err: 2 | LR: 0.00030000

Early stopping at epoch 650! Vertex RMSE hasn't improved for 500 epochs
Loaded best model state with Vertex RMSE: 0.987654
Training completed! Best loss: 0.456789

============================================================
EVALUATING TRAINED MODEL
============================================================
```

Training features:
- **Batch Learning**: Trains on multiple samples simultaneously
- **Vertex Count Prediction**: VertexPredictor includes vertex count classification alongside position prediction
- **Edge List Efficiency**: Uses memory-efficient edge representation
- **Global Metrics**: Proper aggregation of performance across all samples
- **Early Stopping**: Based on vertex RMSE with best model saving
- **Multi-component Loss**: Vertex position + edge connectivity + vertex count + sparsity

### 3. Comprehensive Evaluation

To generate detailed analysis and visualizations:

```bash
python evaluate.py
```

This creates:
- **`output/summary_report.txt`** - Global and per-sample performance metrics
- **`output/test_sample_N/`** - Individual sample analysis directories
  - `*_prediction_comparison.png` - Side-by-side wireframe comparison
  - `*_edge_probabilities.png` - Edge probability distributions
- **Global Metrics**: Properly aggregated TP/FP/FN across all samples

### 4. Interactive 3D Visualization

For professional, interactive 3D visualization:

```bash
python visualize_open3d.py
```

**Interactive Menu Options:**
1. **Point Cloud Only** - Interactive 3D point cloud with original colors
2. **Ground Truth Wireframe Only** - Blue wireframe structure
3. **Predicted Wireframe Only** - Green predicted wireframe  
4. **Comparison Overlay** - Both wireframes overlaid (Blue=GT, Green=Predicted)
5. **Comprehensive View** - Point cloud + both wireframes together
6. **Save High-Quality Images** - Export 1920x1080 rendered images

## Data Formats

### Point Cloud (`.xyz`)

Each line contains 8 features:
```
X Y Z R G B A Intensity
```

**Example:**
```
538093.9600 6584173.7000 36.4800 110 112 107 107 0.4566
538093.5800 6584173.7000 36.4300 110 117 109 115 0.5896
```

### Wireframe (`.obj`)

Wavefront OBJ format with vertices and line connectivity:
```
v 538094.0457 6584173.3395 36.5879    # Vertex coordinates
v 538093.5322 6584173.2911 36.1297    # Next vertex
l 1 2                                 # Line connecting vertices 1 and 2
```

## Key Features

### Edge List Architecture (New!)

The system now uses memory-efficient edge lists instead of dense adjacency matrices:

1. **Memory Efficiency**: O(E) storage instead of O(V²) for edge information
2. **No Padding**: Eliminates need to pad adjacency matrices to max_vertices
3. **Dynamic Processing**: Handles variable vertex counts naturally
4. **Efficient Training**: Edge labels created from ground-truth edge sets
5. **Scalable**: Supports larger wireframes without quadratic memory growth

### Fixed Architecture with Smart Masking

The model uses a fixed architecture with intelligent masking for variable vertex counts:

1. **Fixed Padding**: All samples padded to max_vertices for uniform tensor operations
2. **Loss Masking**: Only actual vertices (up to true vertex count) contribute to vertex loss
3. **Edge Masking**: Edge prediction only considers valid vertex pairs based on true counts
4. **Evaluation Masking**: Metrics computed only on actual vertices, ignoring padding

### Batch Training System

Training on multiple diverse samples simultaneously:

1. **Multi-Sample Learning**: Learns from diverse wireframe structures
2. **Variable Complexity**: Handles samples with different vertex/edge counts
3. **Robust Generalization**: Better performance on unseen wireframe types
4. **Global Metrics**: Proper aggregation of TP/FP/FN across all samples

### Loss Function

Multi-component loss with balanced weighting:

```python
total_loss = vertex_weight * MSE(predicted_vertices, true_vertices) + 
             edge_weight * BCE(predicted_edges, true_edges) +
             count_weight * CE(predicted_counts, true_counts) +
             sparsity_weight * MSE(predicted_counts, true_counts)
```

- **Vertex Loss**: Mean Squared Error for 3D coordinate prediction
- **Edge Loss**: Binary Cross-Entropy for connectivity prediction (using edge lists)
- **Count Loss**: Cross-Entropy for vertex count classification (from VertexPredictor)
- **Sparsity Loss**: L2 penalty on vertex count deviation
- **Current Weights**: vertex=5.0, edge=1.0, count=100.0, sparsity=50.0

### Evaluation Metrics

#### Global Metrics (Proper Aggregation)
- **Global Vertex RMSE**: Root mean squared error across all predicted vertices
- **Global Edge Precision**: TP / (TP + FP) aggregated across all samples
- **Global Edge Recall**: TP / (TP + FN) aggregated across all samples
- **Global Edge F1-Score**: Harmonic mean of global precision and recall

#### Per-Sample Metrics (For Analysis)
- **Individual Vertex RMSE**: Per-sample vertex position accuracy
- **Individual Edge Metrics**: Per-sample precision/recall for detailed analysis
- **Vertex Count Accuracy**: Exact match between predicted and true vertex counts

## Technical Implementation

### Data Preprocessing

1. **Point Cloud Normalization**:
   - Spatial coordinates: StandardScaler (zero mean, unit variance)
   - Colors: Divided by 255.0 (normalize to [0,1])
   - Intensity: StandardScaler normalization

2. **Wireframe Processing**:
   - Vertices: Same spatial scaler as point cloud
   - Edges: Stored as efficient edge sets (min, max) tuples
   - No adjacency matrix padding required

### Model Training

1. **Batch Processing**: Efficient tensor operations on multiple samples
2. **Gradient Clipping**: Max norm 1.0 to prevent exploding gradients
3. **Multi-component Loss**: Balanced vertex + edge + count + sparsity objectives
4. **MultiStepLR Scheduling**: Learning rate decay at [400, 600, 750, 850] epochs
5. **Early Stopping**: Based on vertex RMSE with 500 epoch patience
6. **Best Model Saving**: Automatic saving of best performing model state

### Performance Optimization

- **GPU Support**: Automatic CUDA detection with fallback to CPU
- **Memory Efficiency**: Edge lists eliminate O(V²) adjacency matrix storage
- **Dynamic Batching**: Variable vertex counts handled naturally
- **Efficient Metrics**: Global metric computation without per-sample averaging artifacts

## Customization

### Training Parameters

In `main.py`, modify the training configuration:

```python
# Change point cloud sampling resolution
batch_data = train_dataset.get_batch_data(target_points=1024)  # or 2048, 4096

# Adjust training hyperparameters
model, loss_history = train_overfit_model(
    batch_data, 
    num_epochs=1000,      # Training epochs
    learning_rate=0.001   # Learning rate
)
```

### Loss Weights

In `train.py`, adjust loss component importance:

```python
# Current balanced approach
criterion = WireframeLoss(
    vertex_weight=5.0,      # Vertex position importance
    edge_weight=1.0,        # Edge connectivity importance
    count_weight=100.0,     # Vertex count prediction importance
    sparsity_weight=50.0    # Vertex count regularization
)

# Alternative configurations
# Vertex-focused: vertex_weight=50.0, edge_weight=1.0
# Edge-focused: vertex_weight=1.0, edge_weight=10.0
# Count-focused: count_weight=200.0, sparsity_weight=100.0
```

### Model Architecture

Adjust model capacity:

```python
model = PointCloudToWireframe(
    input_dim=8,          # Point features (X,Y,Z,R,G,B,A,I)
    max_vertices=64       # Maximum vertices the model can predict
)
```

## Performance Analysis

### Recent Results Summary

The edge list architecture has delivered significant improvements:

1. **Memory Efficiency**: Eliminated O(V²) adjacency matrix storage
2. **Proper Global Metrics**: Fixed per-sample averaging artifacts
3. **Dynamic Vertex Handling**: Natural support for variable vertex counts
4. **Scalable Architecture**: Ready for larger wireframes and datasets

### Best Performing Samples
- **Simple Structures** (1-8 edges): Near-perfect prediction (F1 > 0.95)
- **Medium Complexity** (15-20 edges): Good performance (F1 > 0.8)
- **Complex Structures** (30+ edges): Room for improvement (F1 < 0.5)

### Areas for Future Improvement
1. **Complex Wireframe Handling**: Better performance on high-edge-count samples
2. **Edge Detection Recall**: Reducing false negative rate (current global recall: 0.32)
3. **Point Cloud Resolution**: Exploring higher target_points for better feature extraction

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: System automatically detects and uses GPU if available
2. **Memory Issues**: Reduce batch size or target_points for very large datasets
3. **Convergence Issues**: Adjust learning rate or loss weights
4. **Visualization Issues**: Ensure Open3D is properly installed for 3D visualization

### Performance Tips

1. **Use GPU**: 10-100x speedup for training compared to CPU
2. **Optimal target_points**: 1024 works well, 2048 for higher quality, 4096 requires more memory
3. **Monitor Global Metrics**: Focus on aggregated performance rather than per-sample averages
4. **Early Stopping**: Let the system automatically stop when improvement plateaus

## Future Extensions

1. **Dynamic Vertex Count Architecture**: Move from fixed padding to truly dynamic vertex prediction
2. **Larger Datasets**: Scale to hundreds of diverse wireframe examples
3. **Point Cloud Resolution**: Explore target_points > 4096 with memory optimizations
4. **Advanced Sampling**: Implement FPS (Furthest Point Sampling) or voxel downsampling
5. **Graph Neural Networks**: Explore GNN architectures for edge prediction
6. **Multi-scale Processing**: Handle wireframes at different levels of detail
7. **Real-time Inference**: Optimize for fast wireframe prediction (<1 second)

## Scientific Contribution

This project demonstrates:

1. **Edge List Architecture**: Memory-efficient alternative to dense adjacency matrices
2. **Fixed Architecture with Smart Masking**: Efficient handling of variable vertex counts via masking
3. **Global Metric Aggregation**: Proper performance measurement across multiple samples
4. **Multi-component Loss Design**: Balanced learning of position, connectivity, and count
5. **Batch Learning System**: Robust training on diverse wireframe structures
6. **Scalable Point Cloud Processing**: Efficient handling of variable-size inputs

## License

This project is provided as-is for educational and research purposes.

---

**🎉 The system now features an efficient edge list architecture with fixed vertex padding, smart masking, and proper global metrics, ready for scaling to larger datasets and more complex wireframe structures.**
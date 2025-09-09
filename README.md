# 3D Point Cloud to Wireframe Prediction

A PyTorch deep learning system that predicts 3D wireframe structures from point cloud data using an attention-based neural network architecture.

## Overview

The system takes point cloud data (`.xyz` format) and predicts corresponding wireframe structures (`.obj` format) with vertices and edge connectivity.

### Architecture

- **PointNet Encoder**: Extracts global features from 8D point clouds (XYZ + RGBA + intensity)
- **Vertex Predictor**: Predicts 3D vertex positions with existence probabilities
- **Edge Predictor**: Uses multi-head attention to predict connectivity between vertices

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Usage

**Test the system:**
```bash
python test.py
```

**Train the model:**
```bash
python main.py
```

**Evaluate performance:**
```bash
python evaluate.py
```

**Interactive 3D visualization:**
```bash
python visualize/visualize_open3d.py
```

## Data Format

**Point Cloud (`.xyz`)**: `X Y Z R G B A Intensity`
**Wireframe (`.obj`)**: Standard OBJ format with vertices and line elements

## Dependencies

- PyTorch ≥ 1.9.0
- NumPy ≥ 1.21.0
- scikit-learn ≥ 1.0.0
- matplotlib ≥ 3.4.0
- Open3D ≥ 0.15.0
- wandb ≥ 0.12.0

## Performance

The system uses Building3D benchmark metrics including corner precision/recall, edge precision/recall, and angle consistency for evaluation.

## Directory Structure

demo_dataset/
├── train_dataset/ # Training data
├── test_dataset/ # Test data
models/ # Neural network components
losses/ # Loss functions
visualize/ # Visualization tools
output/ # Results and trained models

Training uses W&B for experiment tracking and automatically saves the best model based on validation performance.
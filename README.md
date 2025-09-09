# 3D Point Cloud to Wireframe Prediction

A PyTorch deep learning system that predicts 3D wireframe structures from point cloud data using an attention-based neural network architecture. The system employs Hungarian RMSE for optimal vertex matching and Building3D benchmark metrics for comprehensive evaluation.

## Overview

This system converts 3D point cloud data into structured wireframe representations, enabling applications in building reconstruction, CAD modeling, and architectural analysis. The pipeline takes dense point cloud data (`.xyz` format) and predicts corresponding wireframe structures (`.obj` format) with accurate vertex positions and edge connectivity.

### Architecture

- **PointNet Encoder**: Extracts both global and local features from 8D point clouds (XYZ coordinates + RGBA color + alpha + intensity values)
- **Vertex Predictor**: Predicts 3D vertex positions with existence probabilities using learned global features
- **Edge Predictor**: Uses multi-head attention mechanism to predict connectivity relationships between predicted vertices

The model is designed to handle variable-sized inputs and outputs while maintaining fixed internal representations for efficient training and inference.

## Dataset

### Demo Dataset Structure

The system includes a comprehensive demo dataset with real-world building point clouds and corresponding wireframe annotations:

```
demo_dataset/
├── train_dataset/
│   ├── point_cloud/     # 43 training point cloud files (.xyz)
│   └── wireframe/       # 43 corresponding wireframe files (.obj)
└── test_dataset/
    ├── point_cloud/     # 8 test point cloud files (.xyz)
    └── wireframe/       # 8 corresponding wireframe files (.obj)
```

### Data Specifications

- **Point Cloud Density**: Typically 1000-10000 points per building
- **Wireframe Complexity**: 10-50 vertices and 20-100 edges per structure
- **Coordinate System**: Metric units (meters) with standardized scaling
- **Color Information**: Full RGBA channels plus intensity for material classification

## Input/Output Format

### Input: Point Cloud (`.xyz` files)
Each line represents a single 3D point with 8 attributes:
```
X Y Z R G B A Intensity
```

**Format Specification:**
- `X, Y, Z`: 3D coordinates in meters (float)
- `R, G, B`: RGB color values [0-255] (int)
- `A`: Alpha transparency [0-255] (int) 
- `Intensity`: Point intensity/reflectance [0.0-1.0] (float)

**Example:**
```
2.45 1.23 0.15 255 128 64 255 0.87
1.12 2.34 1.45 200 150 100 255 0.92
...
```

### Output: Wireframe (`.obj` files)
Standard Wavefront OBJ format containing vertices and line elements:

**Format Specification:**
```
v X Y Z          # Vertex coordinates
l v1 v2          # Line element connecting vertices v1 and v2
```

**Example:**
```
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
l 1 2
l 2 3
l 3 4
l 4 1
```

## Evaluation Metrics

### Hungarian RMSE
The system uses Hungarian algorithm-based RMSE for optimal vertex matching between predictions and ground truth:

- **Algorithm**: Scipy's `linear_sum_assignment` finds minimum-cost bipartite matching
- **Cost Matrix**: Euclidean distances between all predicted-true vertex pairs
- **RMSE Calculation**: Root mean squared error on optimally matched vertex pairs
- **Advantages**: Handles variable vertex counts and finds globally optimal assignment

### Building3D Benchmark Metrics
Industry-standard metrics for 3D wireframe evaluation:

**Corner-based Metrics:**
- **ACO (Average Corner Offset)**: Mean distance from predicted to nearest true vertices
- **CP (Corner Precision)**: Fraction of predicted vertices within threshold of true vertices
- **CR (Corner Recall)**: Fraction of true vertices matched by predicted vertices within threshold  
- **C-F1**: Harmonic mean of corner precision and recall

**Edge-based Metrics:**
- **EP (Edge Precision)**: Fraction of predicted edges that match true edges
- **ER (Edge Recall)**: Fraction of true edges correctly predicted
- **E-F1**: Harmonic mean of edge precision and recall

**Threshold Settings:**
- Corner matching threshold: 2.0 meters
- Edge matching: Exact vertex index correspondence required

### Performance Benchmarks
Current system performance on demo dataset:
- **Global Vertex RMSE**: 3.76m
- **Corner F1-Score**: 0.11
- **Edge F1-Score**: 0.09
- **ACO**: 3.86m

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

## Dependencies

- PyTorch ≥ 1.9.0
- NumPy ≥ 1.21.0
- scikit-learn ≥ 1.0.0 (for Hungarian algorithm and preprocessing)
- scipy ≥ 1.7.0 (for linear sum assignment)
- matplotlib ≥ 3.4.0
- Open3D ≥ 0.15.0 (for 3D visualization)
- wandb ≥ 0.12.0 (for experiment tracking)

## Directory Structure

```
wireframe-3d-prediction/
├── demo_dataset/           # Training and test data
│   ├── train_dataset/      # 43 training samples
│   └── test_dataset/       # 8 test samples
├── models/                 # Neural network components
│   ├── PointNetEncoder.py  # Point cloud feature extraction
│   ├── VertexPredictor.py  # Vertex position prediction
│   └── EdgePredictor.py    # Edge connectivity prediction
├── losses/                 # Loss functions for training
├── visualize/              # 3D visualization tools
├── output/                 # Results and trained models
├── main.py                 # Training pipeline
├── test.py                 # Testing and evaluation
├── evaluate.py             # Comprehensive metrics
└── train.py                # Core training functions
```

## Training & Evaluation

- **Training Strategy**: Uses W&B for experiment tracking and hyperparameter optimization
- **Model Selection**: Automatically saves best model based on validation Hungarian RMSE
- **Evaluation**: Comprehensive reporting with both custom metrics and Building3D benchmarks
- **Visualization**: Interactive 3D visualization of predictions vs ground truth

The system provides detailed performance analysis including per-sample breakdowns, error distributions, and comparative visualizations for thorough model assessment.
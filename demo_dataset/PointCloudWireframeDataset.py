import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import logging


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

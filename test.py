import numpy as np
from scipy.spatial.distance import cdist

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
        true_vertices = result['true_vertices']
        
        # Calculate ACO (Average Corner Offset) - proper implementation
        if len(pred_vertices) > 0 and len(true_vertices) > 0:
            # For each predicted vertex, find distance to nearest true vertex
            distances = cdist(pred_vertices, true_vertices)
            min_distances = np.min(distances, axis=1)  # closest true vertex for each prediction
            aco = np.mean(min_distances)
        else:
            aco = float('inf')
        
        # Corner Precision and Recall - proper per-vertex calculation
        corner_threshold = 2.0  # meters - adjust based on your scale
        
        if len(pred_vertices) > 0 and len(true_vertices) > 0:
            # Find matches using Hungarian-style assignment but with threshold
            distances = cdist(pred_vertices, true_vertices)
            
            # For each predicted vertex, check if it has a match within threshold
            pred_has_match = np.min(distances, axis=1) <= corner_threshold
            correctly_predicted_corners = np.sum(pred_has_match)
            
            # For each true vertex, check if it has a match within threshold  
            true_has_match = np.min(distances, axis=0) <= corner_threshold
            correctly_recalled_corners = np.sum(true_has_match)
            
            # Corner Precision: correctly predicted / total predicted
            cp = correctly_predicted_corners / len(pred_vertices) if len(pred_vertices) > 0 else 0
            
            # Corner Recall: correctly recalled / total true
            cr = correctly_recalled_corners / len(true_vertices) if len(true_vertices) > 0 else 0
            
        elif len(pred_vertices) == 0 and len(true_vertices) == 0:
            cp = 1.0  # Perfect match when both empty
            cr = 1.0
        else:
            cp = 0.0  # No match when counts differ
            cr = 0.0
        
        # Corner F1 score
        c_f1 = 2 * cp * cr / (cp + cr) if (cp + cr) > 0 else 0
        
        # Edge metrics (these were already correct)
        ep = result['edge_precision']
        er = result['edge_recall']
        e_f1 = result['edge_f1_score']
        
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
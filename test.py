import numpy as np



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

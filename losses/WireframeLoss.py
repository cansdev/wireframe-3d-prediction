import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class WireframeLoss(nn.Module):
    """
    Combined multi-task loss function for wireframe prediction
    
    This loss combines two different objectives:
    1. Vertex Position Loss: MSE loss for 3D coordinate accuracy
    2. Edge Connectivity Loss: BCE loss for edge existence prediction
    
    The loss is designed to handle variable vertex counts by masking
    and only computing losses on active vertices/edges.
    """
    
    def __init__(self, vertex_weight=1.0, edge_weight=1.0, existence_weight=1.0):
        """
        Initialize the combined wireframe loss
        
        Args:
            vertex_weight (float): Weight for vertex position loss (default: 1.0)
            edge_weight (float): Weight for edge connectivity loss (default: 1.0)
            existence_weight (float): Weight for vertex existence loss (default: 1.0)
        """
        super(WireframeLoss, self).__init__()
        
        # Store loss weights for different components
        self.vertex_weight = vertex_weight
        self.edge_weight = edge_weight
        self.existence_weight = existence_weight
        
        # Initialize loss functions for different components
        self.smooth_l1_loss = nn.SmoothL1Loss()  # For vertex position regression (better than MSE)
        self.bce_loss = nn.BCELoss()             # For edge existence classification
        
    def forward(self, predictions, targets):
        """
        Compute the combined wireframe loss with built-in Hungarian matching
        
        Args:
            predictions (dict): Model predictions containing:
                - vertices: Predicted vertex coordinates (batch_size, max_vertices, 3)
                - existence_probabilities: Vertex existence probabilities (batch_size, max_vertices)
                - edge_probs: Edge existence probabilities (batch_size, num_edges)
                
            targets (dict): Ground truth targets containing:
                - vertices: Target vertex coordinates (batch_size, max_vertices, 3)
                - vertex_existence: Target vertex existence labels (batch_size, max_vertices)
                - edge_labels: Target edge existence labels (batch_size, num_edges)
                - vertex_counts: Target vertex counts (batch_size,)
                
        Returns:
            dict: Dictionary containing individual and total losses
        """
        batch_size = predictions['vertices'].shape[0]
        
        # COMPONENT 1: Vertex Position Loss (Smooth L1 on Hungarian matched vertices)
        pred_vertices = predictions['vertices']      # (batch_size, max_vertices, 3)
        target_vertices = targets['vertices']        # (batch_size, max_vertices, 3)
        target_existence = targets['vertex_existence']  # (batch_size, max_vertices) - binary labels
        vertex_counts = targets['vertex_counts']     # (batch_size,) - actual vertex counts
        
        # Perform Hungarian matching internally
        matched_indices = self._hungarian_matching(predictions, targets)
        
        # Use Hungarian matching for vertex loss computation
        vertex_loss = self._compute_matched_vertex_loss(pred_vertices, target_vertices, matched_indices)
        
        # COMPONENT 2: Vertex Existence Loss (BCE on existence probabilities)
        pred_existence = predictions['existence_probabilities']  # (batch_size, max_vertices)
        existence_loss = self.bce_loss(pred_existence, target_existence.float())
        
        # COMPONENT 3: Edge Connectivity Loss (BCE on predicted edge probabilities)
        pred_edge_probs = predictions['edge_probs']    # (batch_size, num_edges) - probabilities [0,1]
        target_edge_labels = targets['edge_labels']    # (batch_size, num_edges) - binary labels {0,1}
        
        # Handle variable edge counts by masking (edges depend on vertex count)
        if pred_edge_probs.numel() > 0 and target_edge_labels.numel() > 0:
            # Make sure dimensions match (handle cases where prediction and target have different edge counts)
            min_edges = min(pred_edge_probs.shape[1], target_edge_labels.shape[1])
            if min_edges > 0:
                pred_edges_masked = pred_edge_probs[:, :min_edges]      # Truncate to common size
                target_edges_masked = target_edge_labels[:, :min_edges]  # Truncate to common size
                edge_loss = self.bce_loss(pred_edges_masked, target_edges_masked)
            else:
                edge_loss = torch.tensor(0.0, device=pred_vertices.device)  # No edges case
        else:
            edge_loss = torch.tensor(0.0, device=pred_vertices.device)  # Empty predictions case
        
        # FINAL: Combine all loss components with respective weights
        # Combined loss with weighted sum of all components
        total_loss = (self.vertex_weight * vertex_loss +      # Coordinate accuracy
                     self.existence_weight * existence_loss + # Vertex existence accuracy
                     self.edge_weight * edge_loss)            # Connectivity accuracy  
        
        # Return detailed loss breakdown for monitoring and debugging
        return {
            'total_loss': total_loss,          # Combined weighted loss
            'vertex_loss': vertex_loss,        # Smooth L1 loss for vertex positions
            'existence_loss': existence_loss,  # BCE loss for vertex existence
            'edge_loss': edge_loss,            # BCE loss for edge connectivity
        }
    
    def _hungarian_matching(self, predictions, targets):
        """
        Perform Hungarian matching between predictions and targets,
        prioritizing matching "existing" predicted vertices with actual target vertices.
        
        Args:
            predictions (dict): Model predictions
            targets (dict): Ground truth targets
            
        Returns:
            list: List of (pred_indices, target_indices) tuples for each batch element
        """
        batch_size = predictions['vertices'].shape[0]
        max_pred_vertices = predictions['vertices'].shape[1] # max_vertices olarak adlandırıldı
        
        pred_vertices = predictions['vertices']  # (batch_size, max_pred_vertices, 3)
        pred_existence = predictions['existence_probabilities']  # (batch_size, max_pred_vertices)
        
        target_vertices = targets['vertices']  # (batch_size, max_target_vertices, 3) - Varsayım: target'ta da bir max_vertex_count var
        target_vertex_counts = targets['vertex_counts']  # (batch_size,)
        
        matched_indices = []
        
        for batch_idx in range(batch_size):
            actual_target_count = target_vertex_counts[batch_idx].item()
            
            # --- Adım 1: Mevcut Ground Truth Vertexleri İçin Maliyet Matrisi Oluşturma ---
            
            # Tahmin edilen ve gerçek verteksler
            pred_v = pred_vertices[batch_idx]  # (max_pred_vertices, 3)
            pred_e = pred_existence[batch_idx]  # (max_pred_vertices,)
            
            # Yalnızca var olan hedefleri kullan
            target_v_actual = target_vertices[batch_idx, :actual_target_count]  # (actual_target_count, 3)
            
            # Konum Maliyeti (L1)
            cost_vertex_pos = torch.cdist(pred_v, target_v_actual, p=1)  # (max_pred_vertices, actual_target_count)
            
            # Varoluş Maliyeti: Tahmin edilen verteksin var olması isteniyorsa (ground truth ile eşleşiyorsa)
            # cost_existence_match = 1 - pred_e  # pred_e ne kadar yüksekse maliyet o kadar az
            # Veya direkt pred_e'nin 1'e olan uzaklığı:
            cost_existence_match = torch.abs(pred_e.unsqueeze(1) - 1.0) # (max_pred_vertices, 1)
            cost_existence_match = cost_existence_match.expand(-1, actual_target_count) # (max_pred_vertices, actual_target_count)

            # Eşleşen durumlar için toplam maliyet
            cost_to_actual_targets = cost_vertex_pos + cost_existence_match
            
            # --- Adım 2: Eşleşmeyen Tahminler İçin "Boşluk" Maliyeti Ekleme ---
            # Hungarian algoritması genellikle kare bir matris veya en azından satır/sütun sayısının eşit olduğu bir matris bekler
            # veya max_pred_vertices vs actual_target_count olduğunda küçük olan kadar eşleşme yapar.
            # "Doğru olanları seçmek" demek, ground truth'ta karşılığı olmayan tahminlerin de bir maliyeti olması demektir.
            
            # Boşluk sütunları oluştur (yani, ground truth'ta karşılığı olmayan durumlar)
            num_dummy_targets = max_pred_vertices - actual_target_count
            
            # Eğer ground truth'tan daha fazla tahmin varsa ve her tahmin bir yere atanmak zorundaysa
            # veya ground truth'a göre bir eşleşme bulunamayan tahminler için maliyet belirlemek istiyorsak
            if num_dummy_targets > 0:
                # Bir tahminin "boşluk" (ground truth'ta eşleşme yok) ile eşleşme maliyeti
                # Bu, tahmin edilen vertexin var olmamasını teşvik etmelidir.
                # Yani, pred_e ne kadar yüksekse (var olma olasılığı), boşlukla eşleşme maliyeti o kadar yüksek olmalı.
                cost_to_dummy_targets = pred_e.unsqueeze(1) # (max_pred_vertices, 1)
                cost_to_dummy_targets = cost_to_dummy_targets.expand(-1, num_dummy_targets) # (max_pred_vertices, num_dummy_targets)
                
                # Tüm maliyet matrisini birleştir
                full_cost_matrix = torch.cat((cost_to_actual_targets, cost_to_dummy_targets), dim=1) # (max_pred_vertices, max_pred_vertices)
            else:
                # Eğer tahmin sayısı ground truth sayısından az veya eşitse, boşluk sütunlarına gerek yok.
                # Ancak Hungarian hala tüm tahminleri bir hedefle (veya boşlukla) eşleştirmeye çalışır.
                # Burada sadece mevcut hedeflere atama yapılır.
                # Eğer max_pred_vertices < actual_target_count ise, bazı ground truth'lar eşleşmez kalır.
                # Eğer max_pred_vertices == actual_target_count ise, birebir eşleşme aranır.
                full_cost_matrix = cost_to_actual_targets
                # Not: Bu durumda hala max_pred_vertices < actual_target_count olabilir.
                # linear_sum_assignment bu durumda sadece min(M, N) adet eşleşme döndürür.
                # En basiti için matrisi yine de kare yapalım:
                if actual_target_count > max_pred_vertices:
                     # Ground truth'ta daha fazla vertex varsa, matrisi genişletelim
                    padding_needed = actual_target_count - max_pred_vertices
                    # Eşleşmeyen tahminler için büyük bir ceza
                    padding_rows = torch.full((padding_needed, actual_target_count), float('inf'), device=pred_v.device)
                    full_cost_matrix = torch.cat((full_cost_matrix, padding_rows), dim=0)

                elif actual_target_count < max_pred_vertices:
                    # Daha önce num_dummy_targets ile ele aldığımız durum, bu else bloğunda olmamalı.
                    # Bu durumda full_cost_matrix = cost_to_actual_targets olacaktır,
                    # linear_sum_assignment ise min(M, N) = actual_target_count kadar eşleşme bulacaktır.
                    pass # Bu kısım önceki if bloğu ile ele alındı, veya aşağıdaki basit formül.
            
            # Daha basit ve genellikle yeterli bir yaklaşım: Sadece mevcut hedefler ile eşleşme matrisini kullan
            # ve eğer `actual_target_count` < `max_pred_vertices` ise, `linear_sum_assignment` doğal olarak
            # `actual_target_count` kadar eşleşme bulur. Kalan tahminler eşleşmez kalır.
            # "Doğru olanları seçmek" burada, eşleşen tahminlerin olması istenen varoluş değerlerine sahip olmasıdır.
            
            # --- Genel Maliyet Matrisi Oluşturma ---
            # prediction_idx (0..max_pred_vertices-1) -> target_idx (0..actual_target_count-1)
            # VEYA
            # prediction_idx (0..max_pred_vertices-1) -> "Boşluk" (dummy target)
            
            # Ağırlıklandırma ekleyelim, varsayılan olarak 1.0 (ayarlanabilir)
            alpha = 1.0 # Konum maliyeti ağırlığı
            beta = 1.0  # Varoluş maliyeti ağırlığı (match durumunda)
            gamma = 1.0 # Varoluş maliyeti (dummy durumunda)

            # Mevcut hedeflerle eşleşme maliyeti
            cost_to_actual_targets = alpha * cost_vertex_pos + beta * torch.abs(pred_e.unsqueeze(1) - 1.0).expand(-1, actual_target_count)

            # Boşluk hedefleri (dummy targets) ile eşleşme maliyeti
            # Bu maliyet, tahmin edilen vertexin var olmaması gerektiğini teşvik eder.
            # Yani, pred_e ne kadar yüksekse (var olma olasılığı), boşlukla eşleşme maliyeti o kadar yüksek olmalı.
            num_dummy_targets_to_add = max_pred_vertices - actual_target_count
            if num_dummy_targets_to_add > 0:
                cost_to_dummy_targets = gamma * pred_e.unsqueeze(1).expand(-1, num_dummy_targets_to_add)
                final_cost_matrix = torch.cat((cost_to_actual_targets, cost_to_dummy_targets), dim=1)
            else: # Eğer ground truth sayısı tahmin sayısına eşit veya fazlaysa
                  # Bu durumda, sadece actual_target_count kadar eşleşme bulunur.
                  # Kalan ground truth'lar eşleşmez kalır.
                  # Veya matrisi kare yapmak için dummy prediction ekleyebiliriz (daha kompleks).
                final_cost_matrix = cost_to_actual_targets
            
            # Eğer ground truth sayısı tahmin sayısından fazlaysa, matrisi kare yapmak için dummy predictions ekleyebiliriz
            # Bu, bazı ground truth'ların eşleşmediğini gösterir.
            if actual_target_count > max_pred_vertices:
                num_dummy_preds_to_add = actual_target_count - max_pred_vertices
                # Sonsuz maliyetle dummy prediction'lar ekle, böylece ground truth'lar onlarla eşleşmez
                dummy_pred_costs = torch.full((num_dummy_preds_to_add, final_cost_matrix.shape[1]), float('inf'), device=pred_v.device)
                final_cost_matrix = torch.cat((final_cost_matrix, dummy_pred_costs), dim=0)

            # Apply Hungarian algorithm
            cost_matrix_np = final_cost_matrix.detach().cpu().numpy()
            pred_indices, target_indices = linear_sum_assignment(cost_matrix_np)
            
            # Eşleşenlerin sadece gerçek hedeflere atanmış olanlar olduğundan emin ol
            # Dummy hedeflere atanmış olanları filtrele
            valid_match_mask = (target_indices < actual_target_count)
            pred_indices_filtered = pred_indices[valid_match_mask]
            target_indices_filtered = target_indices[valid_match_mask]
            
            matched_indices.append((pred_indices_filtered, target_indices_filtered))
        
        return matched_indices
    
    def _compute_matched_vertex_loss(self, pred_vertices, target_vertices, matched_indices):
        """
        Compute vertex loss using Hungarian matched pairs.
        
        Args:
            pred_vertices: Predicted vertices (batch_size, max_vertices, 3)
            target_vertices: Target vertices (batch_size, max_vertices, 3)
            matched_indices: List of (pred_indices, target_indices) tuples for each batch element
            
        Returns:
            torch.Tensor: Smooth L1 loss on matched vertex pairs
        """
        batch_size = pred_vertices.shape[0]
        device = pred_vertices.device
        
        total_loss = 0.0
        total_matches = 0
        
        for batch_idx in range(batch_size):
            pred_idx, target_idx = matched_indices[batch_idx]
            
            if len(pred_idx) > 0:
                # Get matched predictions and targets
                matched_pred = pred_vertices[batch_idx, pred_idx]  # (num_matches, 3)
                matched_target = target_vertices[batch_idx, target_idx]  # (num_matches, 3)
                
                # Compute Smooth L1 loss for this batch element
                batch_loss = self.smooth_l1_loss(matched_pred, matched_target)
                total_loss += batch_loss * len(pred_idx)  # Weight by number of matches
                total_matches += len(pred_idx)
        
        # Average over total number of matches across all batches
        if total_matches > 0:
            return total_loss / total_matches
        else:
            return torch.tensor(0.0, device=device)

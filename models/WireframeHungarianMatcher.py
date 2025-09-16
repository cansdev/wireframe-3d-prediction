"""
Hungarian matching for wireframe vertex prediction.
Adapted from DETR's HungarianMatcher for 3D vertex matching.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class WireframeHungarianMatcher(nn.Module):
    """Hungarian matcher for wireframe vertex prediction.
    
    This class computes an assignment between predicted vertices and ground truth vertices
    using the Hungarian algorithm to minimize the total matching cost.
    """

    def __init__(self, cost_vertex: float = 1.0, cost_existence: float = 1.0):
        """Creates the matcher

        Params:
            cost_vertex: This is the relative weight of the L1 error of vertex coordinates in the matching cost
            cost_existence: This is the relative weight of the existence probability error in the matching cost
        """
        super().__init__()
        self.cost_vertex = cost_vertex
        self.cost_existence = cost_existence
        assert cost_vertex != 0 or cost_existence != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains:
                 "vertices": Tensor of dim [batch_size, max_vertices, 3] with predicted vertex coordinates
                 "existence_probabilities": Tensor of dim [batch_size, max_vertices] with existence probabilities

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "vertices": Tensor of dim [num_target_vertices, 3] containing the target vertex coordinates
                 "existence": Tensor of dim [num_target_vertices] containing existence labels (1 for existing, 0 for non-existing)

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(max_vertices, num_target_vertices)
        """
        bs, num_queries = outputs["vertices"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_vertices = outputs["vertices"].flatten(0, 1)  # [batch_size * num_queries, 3]
        out_existence = outputs["existence_probabilities"].flatten(0, 1)  # [batch_size * num_queries]

        # Also concat the target vertices and existence
        tgt_vertices = torch.cat([v["vertices"] for v in targets])
        tgt_existence = torch.cat([v["existence"] for v in targets])

        # Compute the L1 cost between vertices
        cost_vertex = torch.cdist(out_vertices, tgt_vertices, p=1)

        # Compute the existence probability cost
        # For existence, we want to minimize the difference between predicted and target existence
        cost_existence = torch.abs(out_existence.unsqueeze(1) - tgt_existence.unsqueeze(0))

        # Final cost matrix
        C = self.cost_vertex * cost_vertex + self.cost_existence * cost_existence
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["vertices"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_wireframe_matcher(cost_vertex=1.0, cost_existence=1.0):
    """Build a wireframe Hungarian matcher with specified costs."""
    return WireframeHungarianMatcher(cost_vertex=cost_vertex, cost_existence=cost_existence)

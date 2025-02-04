import torch

def compute_distances_and_angles_combined(points_tensor):
    """
    Computes pairwise distances and angles between triplets of points in a 3D pose tensor.

    This function extracts geometric features from a set of 3D points across multiple frames:
    1. Computes pairwise Euclidean distances between all points.
    2. Computes angles formed by three points using the cosine rule.

    Args:
        points_tensor (torch.Tensor): A tensor of shape (num_frames, num_points, 3),
                                      where each point has (x, y, z) coordinates.

    Returns:
        torch.Tensor: A tensor of shape (num_frames, num_features), where num_features is 
                      the total number of pairwise distances and angles concatenated together.

    Raises:
        RuntimeError: If `points_tensor` is not a 3D tensor with shape (num_frames, num_points, 3).
        
    """
    if points_tensor.dim() != 3 or points_tensor.size(-1) != 3:
        raise RuntimeError("Input tensor must have shape (num_frames, num_points, 3).")

    num_frames, num_points, _ = points_tensor.size()

    # Precompute all pairs and triplets of indices
    pairs = torch.combinations(torch.arange(num_points), r=2, with_replacement=False)
    triplets = torch.combinations(torch.arange(num_points), r=3, with_replacement=False)

    # Compute pairwise distances
    point_diffs = points_tensor[:, pairs[:, 0]] - points_tensor[:, pairs[:, 1]]  # [num_frames, num_pairs, 3]
    pairwise_distances = torch.norm(point_diffs, dim=2)  # [num_frames, num_pairs]

    # Compute angles between triplets using the cosine rule
    vec1 = points_tensor[:, triplets[:, 0]] - points_tensor[:, triplets[:, 1]]  # [num_frames, num_triplets, 3]
    vec2 = points_tensor[:, triplets[:, 2]] - points_tensor[:, triplets[:, 1]]  # [num_frames, num_triplets, 3]
    dot_products = torch.sum(vec1 * vec2, dim=2)  # [num_frames, num_triplets]
    norms = torch.norm(vec1, dim=2) * torch.norm(vec2, dim=2)  # [num_frames, num_triplets]
    cos_angles = dot_products / (norms + 1e-8)  # Add epsilon to avoid division by zero

    # Concatenate distances and angles
    combined_features = torch.cat([pairwise_distances, cos_angles], dim=1)  # [num_frames, num_features]
    return combined_features
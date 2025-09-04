import torch
import numpy as np
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def load_json_as_numpy(json_file):
    """
    Load a JSON file and convert it to numpy array.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        np.ndarray: The loaded data as a numpy array.
    """
    with open(json_file, "r") as file:
        data = json.load(file)
    return np.array(data)


def process_action_data(base_path, action_name):
    """
    Process data for a specific action by loading the corresponding Excel and JSON files,
    adding pose data, and splitting into train, validation, and test sets.

    Args:
        base_path (str): The base directory containing the files.
        action_name (str): The name of the action (e.g., 'squat', 'deadlift', 'lunges').

    Returns:
        tuple: train_df, val_df, test_df DataFrames.
    """
    # Load the Excel file
    excel_file = f"{action_name}.xlsx"
    df = pd.read_excel(os.path.join(base_path, excel_file))

    # Load the JSON files for front and lateral poses
    front_pose_file = f"front_pose_{action_name}.json"
    lat_pose_file = f"lat_pose_{action_name}.json"

    front_pose_array = load_json_as_numpy(os.path.join(base_path, front_pose_file))
    lat_pose_array = load_json_as_numpy(os.path.join(base_path, lat_pose_file))

    # Ensure `front_pose` and `lat_pose` columns exist
    if "front_pose" not in df.columns:
        df["front_pose"] = None
    if "lat_pose" not in df.columns:
        df["lat_pose"] = None

    # Assign the loaded arrays to the DataFrame if lengths match
    if len(front_pose_array) == len(df) and len(lat_pose_array) == len(df):
        df["front_pose"] = list(front_pose_array)
        df["lat_pose"] = list(lat_pose_array)
    else:
        raise ValueError(
            "The length of the loaded arrays does not match the DataFrame."
        )

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Define split ratios
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

    # Calculate the number of samples for each set
    total_samples = len(df)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    # Split the DataFrame
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size : train_size + val_size]
    test_df = df.iloc[train_size + val_size :]

    return train_df, val_df, test_df


def compute_pairwise_distances(points_tensor):
    """
    Compute pairwise distances between 33 pose landmarks for each frame.

    Args:
        points_tensor (torch.Tensor): Tensor of shape [num_frames, 33, 3],
                                     where each row is a point (x, y, z) for each frame.

    Returns:
        torch.Tensor: Tensor of shape [num_frames, 528], containing pairwise distances for each frame.
    """
    num_frames, num_points, _ = points_tensor.size()

    # Generate index pairs for the upper triangle of a matrix (excluding diagonal)
    pairs = torch.combinations(
        torch.arange(num_points), r=2, with_replacement=False
    )  # Shape: [528, 2]

    # Gather the coordinates for each pair of points
    point1 = points_tensor[:, pairs[:, 0]]  # Shape: [num_frames, 528, 3]
    point2 = points_tensor[:, pairs[:, 1]]  # Shape: [num_frames, 528, 3]

    # Compute pairwise Euclidean distances
    pairwise_distances = torch.norm(point1 - point2, dim=2)  # Shape: [num_frames, 528]

    return pairwise_distances


def compute_distances_and_angles_combined(points_tensor):
    """
    Compute and concatenate pairwise distances and angles between every three points.

    Args:
        points_tensor (torch.Tensor): Tensor of shape [num_frames, 33, 3],
                                     where each row is a point (x, y, z) for each frame.

    Returns:
        torch.Tensor: Tensor of shape [num_frames, 5984],
                      containing pairwise distances and angles for each frame.
    """
    num_frames, num_points, _ = points_tensor.size()

    # Precompute all pairs and triplets of indices
    pairs = torch.combinations(torch.arange(num_points), r=2, with_replacement=False)
    triplets = torch.combinations(torch.arange(num_points), r=3, with_replacement=False)

    # Compute pairwise distances
    point_diffs = (
        points_tensor[:, pairs[:, 0]] - points_tensor[:, pairs[:, 1]]
    )  # [num_frames, num_pairs, 3]
    pairwise_distances = torch.norm(point_diffs, dim=2)  # [num_frames, num_pairs]

    # Compute angles between triplets
    vec1 = (
        points_tensor[:, triplets[:, 0]] - points_tensor[:, triplets[:, 1]]
    )  # [num_frames, num_triplets, 3]
    vec2 = (
        points_tensor[:, triplets[:, 2]] - points_tensor[:, triplets[:, 1]]
    )  # [num_frames, num_triplets, 3]
    dot_products = torch.sum(vec1 * vec2, dim=2)  # [num_frames, num_triplets]
    norms = torch.norm(vec1, dim=2) * torch.norm(
        vec2, dim=2
    )  # [num_frames, num_triplets]
    cos_angles = dot_products / (norms + 1e-8)  # Add epsilon to avoid division by zero

    # Concatenate distances and angles
    combined_features = torch.cat(
        [pairwise_distances, cos_angles], dim=1
    )  # [num_frames, 5984]
    return combined_features


def load_and_process_data(base_path, actions=["squat", "deadlift", "lunges"]):
    """
    Load and process data for multiple actions and combine them into train, val, and test sets.

    Args:
        base_path (str): Base path to the dataset.
        actions (list): List of action names to process.

    Returns:
        tuple: train_df, val_df, test_df - Combined DataFrames for all actions.
    """
    # Process data for each action and store results in dictionaries
    splits = {action: process_action_data(base_path, action) for action in actions}

    # Concatenate and shuffle DataFrames for each split
    train_df = (
        pd.concat([splits[action][0] for action in actions])
        .sample(frac=1, random_state=1)
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat([splits[action][1] for action in actions])
        .sample(frac=1, random_state=1)
        .reset_index(drop=True)
    )
    test_df = (
        pd.concat([splits[action][2] for action in actions])
        .sample(frac=1, random_state=1)
        .reset_index(drop=True)
    )

    return train_df, val_df, test_df

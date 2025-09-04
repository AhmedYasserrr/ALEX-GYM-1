import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from ..utils.data_processing import compute_distances_and_angles_combined


class PreprocessedPoseVideoDataset(Dataset):
    """
    Dataset class for loading preprocessed pose and video data.
    """

    def __init__(self, df, num_frames, preprocessed_dir, feature_count=6):
        """
        Initialize the dataset.

        Args:
            df (pd.DataFrame): DataFrame containing video metadata
            num_frames (int): Number of frames to load per video
            preprocessed_dir (str): Directory containing preprocessed frames
            feature_count (int): Number of features/criteria to evaluate
        """
        self.df = df
        self.num_frames = num_frames
        self.preprocessed_dir = preprocessed_dir
        self.feature_count = feature_count

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample

        Returns:
            tuple: Tuple containing the video frames, pose data, and labels
        """
        row = self.df.iloc[idx]

        # Get frontal and lateral video identifiers
        num_video_frontal = row["Num Video Frontal"]
        num_video_lateral = row["Num Video Lateral"]
        num_idx = row["NumIdx"]
        action = row["Action"]

        # Load preprocessed frames
        frontal_frames = self._load_preprocessed_frames(
            num_video_frontal, action, num_idx
        )
        lateral_frames = self._load_preprocessed_frames(
            num_video_lateral, action, num_idx
        )

        # Normalize images
        image_frontal = (
            torch.tensor(frontal_frames, dtype=torch.float32).permute(0, 3, 1, 2)
            / 255.0
        )
        image_lateral = (
            torch.tensor(lateral_frames, dtype=torch.float32).permute(0, 3, 1, 2)
            / 255.0
        )

        label_class = torch.tensor(row["class"], dtype=torch.long)
        ratings = self._process_ratings(row)
        ratings = (
            torch.tensor(ratings, dtype=torch.float32) if ratings is not None else None
        )

        # Extract pose landmarks from video frames
        pose_landmarks_frontal = row["front_pose"]
        pose_landmarks_lateral = row["lat_pose"]
        pose_landmarks_tensor_frontal = torch.tensor(pose_landmarks_frontal).float()
        pose_landmarks_tensor_lateral = torch.tensor(pose_landmarks_lateral).float()
        pose_frontal = compute_distances_and_angles_combined(
            pose_landmarks_tensor_frontal
        )
        pose_lateral = compute_distances_and_angles_combined(
            pose_landmarks_tensor_lateral
        )

        return (
            image_frontal,
            image_lateral,
            pose_frontal,
            pose_lateral,
            label_class,
            ratings,
        )

    def _load_preprocessed_frames(self, num_video, action, num_idx):
        """
        Load preprocessed frames from disk.

        Args:
            num_video (int): Video identifier
            action (str): Action type
            num_idx (int): Index of the video

        Returns:
            np.ndarray: Stacked frames of the video
        """
        frames = []
        for i in range(1, self.num_frames + 1):
            # Construct the preprocessed file path
            file_name = f"{num_video}_idx_{num_idx}_{i}.npy"
            file_path = os.path.join(self.preprocessed_dir, action, file_name)

            # Load the preprocessed .npy file
            frame = np.load(file_path)
            frames.append(frame)
        return np.stack(frames, axis=0)

    def _process_ratings(self, row):
        """
        Process the ratings/criteria from the DataFrame row.

        Args:
            row (pd.Series): DataFrame row

        Returns:
            list: Binary ratings for each feature
        """
        relevant_columns = [
            col for col in row.index if col.endswith("F") or col.endswith("L")
        ]
        if not relevant_columns:
            return None

        scores = row[relevant_columns].values
        thresholded_scores = np.where(scores >= 0.5, 1, 0)
        return thresholded_scores.tolist()


def create_dataloader(
    df, num_frames, preprocessed_dir, batch_size=16, shuffle=True, num_workers=4
):
    """
    Create a DataLoader for the dataset.

    Args:
        df (pd.DataFrame): DataFrame containing video metadata
        num_frames (int): Number of frames to load per video
        preprocessed_dir (str): Directory containing preprocessed frames
        batch_size (int): Batch size for the DataLoader
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading

    Returns:
        DataLoader: PyTorch DataLoader for the dataset
    """
    dataset = PreprocessedPoseVideoDataset(df, num_frames, preprocessed_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Optimize for GPU training
    )

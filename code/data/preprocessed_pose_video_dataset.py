import torch
from torch.utils.data import Dataset
import os
import numpy as np
from ..utils.pose_utils import compute_distances_and_angles_combined

class PreprocessedPoseVideoDataset(Dataset):
    """
    A PyTorch Dataset for loading preprocessed video frames and corresponding pose landmarks.
    
    This dataset assumes that video frames are preprocessed and stored as `.npy` files, 
    and it can handle both frontal and lateral video views. The dataset also includes 
    pose landmark features extracted from the video frames, along with associated action labels 
    and ratings for each sample.

    Args:
        df (pandas.DataFrame): DataFrame containing the metadata for each sample, 
                               including the video identifiers, action types, and pose landmarks.
        num_frames (int): The number of frames to be loaded per video (e.g., 30 frames).
        preprocessed_dir (str): Directory where the preprocessed `.npy` frames are stored.

    Attributes:
        df (pandas.DataFrame): Stores the input DataFrame with metadata and annotations.
        num_frames (int): The number of frames to be processed per video.
        preprocessed_dir (str): The directory path where preprocessed video frames are stored.
    """
    
    def __init__(self, df, num_frames, preprocessed_dir):
        """
        Initializes the dataset by storing the provided DataFrame, number of frames, 
        and directory for preprocessed frames.

        Args:
            df (pandas.DataFrame): Metadata DataFrame with video identifiers and pose data.
            num_frames (int): Number of frames to be loaded for each video sample.
            preprocessed_dir (str): Directory containing preprocessed frames.
        """
        self.df = df
        self.num_frames = num_frames
        self.preprocessed_dir = preprocessed_dir

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves a single sample (video frames, pose landmarks, and labels) from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: A tuple containing the following elements:
                - image_frontal (torch.Tensor): Preprocessed frontal video frames (normalized).
                - image_lateral (torch.Tensor): Preprocessed lateral video frames (normalized).
                - pose_frontal (torch.Tensor): Pose features for the frontal view (distances and angles).
                - pose_lateral (torch.Tensor): Pose features for the lateral view (distances and angles).
                - label_class (torch.Tensor): Action class label.
                - ratings (torch.Tensor or None): Processed ratings (if available).
        """
        row = self.df.iloc[idx]

        # Get frontal and lateral video identifiers
        num_video_frontal = row['Num Video Frontal']
        num_video_lateral = row['Num Video Lateral']
        num_idx = row['NumIdx']
        action = row['Action']

        # Load preprocessed frames
        frontal_frames = self._load_preprocessed_frames(num_video_frontal, action, num_idx)
        lateral_frames = self._load_preprocessed_frames(num_video_lateral, action, num_idx)

        # Normalize images
        image_frontal = torch.tensor(frontal_frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        image_lateral = torch.tensor(lateral_frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

        label_class = torch.tensor(row['class'], dtype=torch.long)
        ratings = self._process_ratings(self.df, row)
        ratings = torch.tensor(ratings, dtype=torch.float32) if ratings is not None else None

        # Extract pose landmarks from video frames
        pose_landmarks_frontal = row['front_pose']
        pose_landmarks_lateral = row['lat_pose']
        pose_landmarks_tensor_frontal = torch.tensor(pose_landmarks_frontal).float()
        pose_landmarks_tensor_lateral = torch.tensor(pose_landmarks_lateral).float()
        pose_frontal = compute_distances_and_angles_combined(pose_landmarks_tensor_frontal)
        pose_lateral = compute_distances_and_angles_combined(pose_landmarks_tensor_lateral)

        return (image_frontal, image_lateral, pose_frontal, pose_lateral, label_class, ratings)

    def _load_preprocessed_frames(self, num_video, action, num_idx):
        """
        Loads preprocessed video frames from `.npy` files.

        Args:
            num_video (int): Identifier for the video.
            action (str): The action name corresponding to the video.
            num_idx (int): Index of the sample within the action.
        
        Returns:
            np.ndarray: A stack of frames (shape: [num_frames, height, width, channels]).
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

    def _process_ratings(self, df, row):
        """
        Processes the ratings of an action based on certain criteria (e.g., thresholding).

        Args:
            df (pandas.DataFrame): DataFrame containing the ratings columns.
            row (pandas.Series): A specific row containing the ratings to process.

        Returns:
            list: List of thresholded ratings (0 or 1).
        """
        relevant_columns = [col for col in df.columns if col.endswith('F') or col.endswith('L')]
        scores = row[relevant_columns].values
        thresholded_scores = np.where(scores >= 0.5, 1, 0)
        return thresholded_scores.tolist()

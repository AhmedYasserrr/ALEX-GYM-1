import os
from data.preprocessed_pose_video_dataset import PreprocessedPoseVideoDataset
from torch.utils.data import DataLoader

def create_dataloader(df, num_frames,base_path, batch_size=16, shuffle=True):
    """
    Creates a DataLoader for the PreprocessedPoseVideoDataset, enabling easy batch loading 
    of preprocessed video frames and pose landmarks for training or evaluation.

    This function wraps the dataset into a DataLoader, which handles batching, shuffling,
    and other data-loading functionalities that are critical during model training.

    Args:
        df (pandas.DataFrame): The DataFrame containing metadata for the dataset, including 
                               identifiers for video files, pose landmarks, and corresponding actions.
        num_frames (int): The number of frames per video clip to be loaded.
        batch_size (int, optional): The batch size for training or evaluation. Default is 16.
        shuffle (bool, optional): Whether to shuffle the data at the beginning of each epoch. 
                                   Default is True.

    Returns:
        torch.utils.data.DataLoader: The DataLoader instance that batches the data and 
                                     loads it efficiently during training or evaluation.

    """
    dataset = PreprocessedPoseVideoDataset(df, num_frames, preprocessed_dir=os.path.join(base_path, 'preprocessed_images'))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True           
    )

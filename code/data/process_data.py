import os
import json
import numpy as np
import pandas as pd

def process_action_data(base_path, action_name):
    """
    Processes data for a specific gym exercise action by loading the corresponding Excel and JSON files,
    adding pose data, and splitting the data into training, validation, and test sets.

    This function performs the following steps:
    1. Loads the Excel file containing metadata and annotations for the specified action.
    2. Loads the JSON files containing pose data for the frontal and lateral views.
    3. Adds the pose data to the DataFrame as new columns.
    4. Shuffles the DataFrame to ensure randomness.
    5. Splits the DataFrame into training, validation, and test sets based on predefined ratios.

    Args:
        base_path (str): The base directory path where the Excel and JSON files are stored.
        action_name (str): The name of the gym exercise action to process (e.g., 'squat', 'deadlift', 'lunges').

    Returns:
        tuple: A tuple containing three DataFrames:
            - train_df: The training set (70% of the data).
            - val_df: The validation set (15% of the data).
            - test_df: The test set (15% of the data).

    Raises:
        ValueError: If the length of the pose data arrays does not match the length of the DataFrame.
        
    """
    # Load the Excel file
    excel_file = f"{action_name}.xlsx"
    df = pd.read_excel(os.path.join(base_path, excel_file))

    # Load the JSON files for front and lateral poses
    front_pose_file = f"front_pose_{action_name}.json"
    lat_pose_file = f"lat_pose_{action_name}.json"

    def load_json_as_numpy(json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
        return np.array(data)

    front_pose_array = load_json_as_numpy(os.path.join(base_path, front_pose_file))
    lat_pose_array = load_json_as_numpy(os.path.join(base_path, lat_pose_file))

    # Ensure `front_pose` and `lat_pose` columns exist
    if 'front_pose' not in df.columns:
        df['front_pose'] = None
    if 'lat_pose' not in df.columns:
        df['lat_pose'] = None

    # Assign the loaded arrays to the DataFrame if lengths match
    if len(front_pose_array) == len(df) and len(lat_pose_array) == len(df):
        df['front_pose'] = list(front_pose_array)
        df['lat_pose'] = list(lat_pose_array)
    else:
        raise ValueError("The length of the loaded arrays does not match the DataFrame.")

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
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    return train_df, val_df, test_df

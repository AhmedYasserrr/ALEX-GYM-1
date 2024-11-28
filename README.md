#Vision_GYM_Research

## Table of Contents
1. [Dataset Overview](#dataset-overview)
   - [Dataset Structure](#dataset-structure)

## Dataset Overview
Our dataset was collected manually and consists of videos capturing specific actions, such as deadlifts, lunges, and squats. Each video contains 4–6 repetitions of the action. To facilitate the evaluation of individual repetitions, we preprocessed the dataset by splitting each video into five sub-videos, corresponding to the average number of repetitions. Each sub-video was further divided into 16 frames to standardize the input.

### Dataset Structure
The dataset is organized into three folders:
- `Deadlift_Frames`
- `Lunges_Frames`
- `Squat_Frames`

The frames within each folder follow a consistent naming convention:  
`{video_idx}_idx_{exercise_idx}_{frame_idx}`  
- **`video_idx`**: Index of the original video (odd numbers for the frontal view, even numbers for the lateral view).  
- **`exercise_idx`**: The specific exercise type.  
- **`frame_idx`**: The frame number within the sub-video.

Here is our dataset: [Dataset Link](https://drive.google.com/drive/u/2/folders/1xeKXsh54_ezwuvA4XI9X39tOUlEf9GwM)

# ALEX-GYM-1 : A Novel Dataset and Hybrid 3D Pose Vision Model for Automated Exercise Evaluation

## Overview

ALEX-GYM-1 is a dataset and hybrid model designed for automated gym exercise evaluation. It combines synchronized multi-view videos (frontal and lateral) with detailed biomechanical annotations for exercises like Squats, Lunges, and Single-Leg Romanian Deadlifts. The dataset supports both pose-based and vision-based analysis, enabling accurate and efficient exercise quality assessment.

The hybrid model integrates:
- **Vision-based features**: Extracted using a 3D ResNet architecture.
- **Pose-based features**: Engineered from 3D skeletal keypoints.

This approach achieves high accuracy in classifying exercise performance and detecting errors.

## Dataset Highlights

- **Exercises**: Squats (295 videos), Lunges (106 videos), Deadlifts (269 videos).
- **Participants**: 45 individuals (diverse age and gender groups).
- **Annotations**: Biomechanical criteria for each exercise.
- **Data**: Includes video frames, pose keypoints (JSON), and metadata (Excel).

## Model Performance

The hybrid model outperforms single-modality approaches:
- **Pose-based model**: Focuses on joint relationships.
- **Vision-based model**: Captures spatio-temporal dynamics.
- **Hybrid model**: Combines both for superior accuracy.

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Download the dataset and organize it as follows:
```
ALEX-GYM-1/
├── Squat_frames/
├── Deadlift_frames/
├── Lunges_frames/
├── squat.xlsx
├── deadlift.xlsx
├── lunges.xlsx
├── front_pose_squat.json
├── front_pose_deadlift.json
├── front_pose_lunges.json
├── lat_pose_squat.json
├── lat_pose_deadlift.json
└──  lat_pose_lunges.json
```


## Results

The hybrid model achieves:
- **Squats**: Hamming Loss 0.0259
- **Deadlifts**: Hamming Loss 0.0488
- **Lunges**: Hamming Loss 0.0756

## Acknowledgment

This work is funded by the Science and Technology Development Fund (STDF), Egypt, under project ID 51399.

--- 

This work is accepted at the ICINCO 2025 conference. The authors are:

- **Ahmed Hassan**
- **Abdelaziz Essam**
- **Ahmed Yasser**
- **Prof. Walid Gomaa**

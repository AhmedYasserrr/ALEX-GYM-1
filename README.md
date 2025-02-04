# ALEX-GYM-1 : A Novel Dataset and Hybrid 3D Pose Vision Model for Automated Exercise Evaluation

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Architecture](#model-architecture)
4. [Results](#results)
5. [Project Setup](#project-setup)
6. [Running the Experiment](#running-the-experiment)

The increasing use of automated systems for exercise monitoring has the potential to revolutionize fitness and rehabilitation. Accurate and efficient exercise evaluation, however, requires the integration of multiple data sources, including visual cues and human pose information. In this context, the ALEX-GYM-1 dataset and the associated hybrid 3D pose vision model aim to address the challenge of automated exercise evaluation.

ALEX-GYM-1 is a novel dataset comprising videos of individuals performing gym exercises from two viewpoints: frontal and lateral. The dataset captures exercises such as Squats, Lunges, and Single-Leg Romanian Deadlifts, making it versatile for evaluating different types of physical activities. Each video is meticulously annotated with key points from 3D pose estimation to support detailed analysis of human motion.

The project presents a hybrid model that combines both vision-based and pose-based features to assess exercise quality. By leveraging a 3D ResNet architecture for vision and a VCOACH-inspired network for pose data, this model aims to accurately classify exercise performance and detect potential errors in real-time.

The following sections provide an in-depth look at the dataset, preprocessing steps, the model architecture, training, and results.

## 1. Dataset Overview

The ALEX-GYM-1 dataset consists of videos capturing individuals performing three fundamental gym exercises: Squats, Lunges, and Single-Leg Romanian Deadlifts. The videos are recorded from two perspectives: frontal and lateral views. The dataset includes diverse participants with various ages and sexes, ensuring that the model can generalize to different body types and demographics.

### Demographic Breakdown
| Category                     | Count |
|------------------------------|-------|
| Female                       | 12    |
| Male                         | 33    |
| **Age Groups**               |       |
| Children (8–12 years)         | 4     |
| Teenagers (13–19 years)       | 7     |
| Young Adults (20–35 years)    | 31    |
| Adults (36–54 years)          | 3     |

The dataset includes 670 total videos distributed as follows:
- **Squats**: 295 videos
- **Lunges**: 106 videos
- **Single-Leg Romanian Deadlifts**: 269 videos

Each video is meticulously annotated based on specific biomechanical features. For each frame, the dataset provides several key information components:
- **Video frames**: Each exercise is recorded as a sequence of video frames, which are key for visual analysis.
- **Pose features**: Pose keypoints are extracted from each frame to capture the human body’s joint positions and help evaluate posture and movement.
- **Excel files**: The dataset includes an Excel sheet with additional features like temporal sequences, repetitions, and any additional annotated labels for further analysis.

The data is organized in a way that supports both **pose-based analysis** and **visual-based analysis**, enabling a hybrid model approach.

## 2. Data Preprocessing

Data preprocessing plays a vital role in preparing the ALEX-GYM-1 dataset for accurate model training. The dataset undergoes two main stages: **General Data Preprocessing** and **Pose Data Preprocessing**. This allows the raw data to be standardized and aligned, ensuring the models can focus on learning from the most relevant features.

### 2.1 General Data Preprocessing
In this stage, we focus on normalizing and structuring the video data. Each video is sampled at **16 frames per perspective** for both the **frontal** and **lateral** views. This consistent frame sampling ensures temporal consistency across the dataset. The videos are first converted into frames, and every frame is processed independently for pose extraction.

Each frame in the dataset is processed through **MediaPipe’s 3D Pose Estimation** model. This model detects 33 key points corresponding to different body joints. The 3D coordinates of these points are recorded for each frame, enabling a detailed analysis of the subject’s body position and movement. After pose extraction, the pose data is saved in **JSON files** corresponding to each video, where each frame’s pose keypoints are stored in a structured format.

Additionally, an **Excel file** is generated for each video containing feature data for each frame. This Excel sheet includes information about the frames' labels, repetitions, and annotations regarding exercise quality (e.g., whether the exercise was performed correctly or incorrectly). This tabular data allows for a deeper dive into the biomechanical features, including joint angles, and provides metadata for training evaluation.

### 2.2 Pose Data Preprocessing
The next step involves the refinement of pose data to enhance the model’s performance. To avoid the inclusion of irrelevant positional information, the absolute positions of the joints are discarded. Instead, we focus on the **spatial relationships between keypoints**. These relationships are critical for assessing movement patterns while being invariant to body scale or position.

#### Pairwise Distance Calculation

A key feature for pose analysis is calculating the **pairwise Euclidean distance** between all pairs of keypoints. Given two keypoints, $ p_i = (x_i, y_i, z_i) $ and $ p_j = (x_j, y_j, z_j) $, the Euclidean distance is calculated as:

$$
d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2}
$$

The dataset contains **33 keypoints** (corresponding to the human body joints). Thus, the total number of unique pairwise distances is:

$$
\text{Total Pairwise Distances} = \frac{33 \times 32}{2} = 528
$$

These pairwise distances capture how the positions of different joints relate to each other and can be indicative of correct or incorrect exercise execution.

#### Pose Angle Calculation

In addition to distances, **angles** between three keypoints are also used. The angle $ \theta_{ijk} $ between three keypoints $ p_i, p_j, p_k $ is computed using the dot product formula:

$$
\cos(\theta_{ijk}) = \frac{(p_i - p_j) \cdot (p_k - p_j)}{\|p_i - p_j\| \|p_k - p_j\|}
$$

The total number of unique pose angle combinations is:

$$
\text{Total Pose Angles} = \binom{33}{3} = 5456
$$

After calculating these pairwise distances and angles, they are concatenated to form a comprehensive set of **5984 features per frame**.

For further details on the dataset post-preprocessing, you can access the processed data through [this link](https://drive.google.com/drive/u/1/folders/124MjpOoon8haYdXwveQAK-hSOoyS-Ivv).

## 3. Model Architecture

The proposed methodology employs two primary deep learning models to evaluate exercise performance using both **video data** and **pose data**. The combination of these models allows for a comprehensive evaluation of exercise quality.

### 3.1 Vision-Based Model
The Vision-Based Model processes the video frames using a **3D ResNet18** architecture. This model is designed to capture the spatial and temporal features of the exercise sequences. In this model, **spatial data** (height and width) is treated as two dimensions, while **temporal data** (frame sequences) is treated as the third dimension. Each video sequence is composed of **16 frames** per perspective, with frames from both frontal and lateral views being processed separately.

The outputs of both perspectives are concatenated and passed through **fully connected layers** to produce binary labels indicating whether the exercise criterion (correct/incorrect) is met.

#### ResNet Block
The **3D ResNet Block** performs feature extraction, with the first two layers being frozen to prevent overfitting during training. The final output provides binary classification labels, which indicate whether the exercise was performed correctly or incorrectly.

### 3.2 Pose-Based Model
The Pose-Based Model processes the 3D pose features derived from each frame. Each pose is represented by **5984 features**, which include both pairwise distances and pose angles. The model is based on the **VCOACH architecture**, which incorporates a series of **2D convolutional layers**, **residual blocks**, and **GRU layers** to capture the temporal dependencies inherent in the exercise sequences.

After processing, the outputs from the Pose-Based Model are passed through **fully connected layers** for final classification.

#### Pose Block Architecture
Each Pose Block consists of:
- **2D convolutional layers** for feature extraction
- **Residual blocks** for enhancing learning capabilities
- **GRU layers** for modeling temporal dependencies across the frames

### 3.3 Merging of Vision and Pose Features
Once both the Vision-Based and Pose-Based Models have been trained, their outputs are merged. The feature vectors of both models (512-dimensional for each model) are concatenated, and the merged features are passed through a **fully connected layer** for the final classification.

The integration of both video and pose data ensures that the model benefits from the complementary strengths of each modality, providing a more robust and accurate exercise evaluation.

### 3.4 Training and Evaluation

The **ALEX-GYM-1 dataset** was split into training, validation, and test sets in a 70:15:15 ratio. Both the Pose-based and Vision-based models were trained independently using the **Adam optimizer** with a learning rate of **5e-4** and a batch size of **16**. Unstable behavior was noticed at higher learning rates such as **8e-4**, so the lower rate was chosen. A **learning rate scheduler**, specifically **ReduceLROnPlateau**, was employed to adjust the learning rate by monitoring the validation loss. If no improvement was observed for 5 consecutive epochs, the learning rate was reduced by a factor of 0.5.

Additionally, **early stopping** was applied with a patience of 10 epochs based on validation accuracy to prevent overfitting. Once the individual models achieved optimal performance, they were integrated into the combined architecture and fine-tuned.

This training process resulted in high classification accuracy for all three exercise types, demonstrating the efficacy of the proposed methodology.

## 4. Results

The evaluation of the proposed models was conducted using **Hamming Loss** as the primary metric. Hamming Loss is suitable for multi-label classification tasks, as it measures the fraction of incorrectly predicted labels relative to the total number of labels. The classification task involves determining whether each feature in the exercise was performed correctly or not based on predefined criteria. Lower Hamming Loss values indicate higher accuracy in classifying and assessing exercise execution.

The **Hamming Loss (HL)** is defined as:

$$
HL = \frac{1}{N \cdot L} \sum_{i=1}^{N} \sum_{j=1}^{L} I(y_{ij} \neq \hat{y}_{ij})
$$

Where:
- \( N \) is the number of samples,
- \( L \) is the number of labels,
- \( y_{ij} \) is the true label for the \( i \)-th sample and \( j \)-th feature,
- \( \hat{y}_{ij} \) is the predicted label for the \( i \)-th sample and \( j \)-th feature,
- \( I(\cdot) \) is the indicator function, which returns **1** if the condition is true and **0** otherwise.

The performance of three distinct models—**Pose-based**, **Vision-based**, and **Merged**—was evaluated across three gym exercises: **Squat**, **Deadlift**, and **Lunge**. 

- The **Pose-based model** relied on joint distances and angles extracted using **MediaPipe**.
- The **Vision-based model** utilized spatio-temporal features extracted from 16-frame video sequences via a **3D CNN**.
- The **Merged model** integrated both Pose-based and Vision-based inputs through a fully connected layer architecture to enhance classification accuracy and movement evaluation.

The Hamming Loss values for each model across the three exercises are summarized in Table II.

| **Exercise** | **Pose-Based Model** | **Vision-Based Model** | **Merged Model** |
|--------------|----------------------|------------------------|------------------|
| **Squat**    | 0.1222               | 0.0519                 | 0.0259           |
| **Deadlift** | 0.1610               | 0.0683                 | 0.0634           |
| **Lunges**   | 0.2605               | 0.1176                 | 0.0840           |

The results, summarized in Table II, demonstrate that the **Merged model** consistently outperforms both the **Pose-based** and **Vision-based** models across all exercises. This model achieved the lowest Hamming Loss in all three exercises, indicating superior classification accuracy and better overall movement assessment. The ability of the Merged model to combine Pose-based and Vision-based features likely contributes to its enhanced performance, as it accounts for both body postures and movement patterns.

For those interested in experimenting with the models, the pre-trained weights for the Pose-based, Vision-based, and Merged models can be downloaded from the following link:[Download Pre-Trained Models](https://drive.google.com/drive/u/1/folders/1-gMrjWqwFsm77gIBTRxaZ8z_6-FMCPcc)

## 5. Project Setup
To begin, ensure that your environment is set up correctly:

### Step 1: Install Required Packages

Clone the repository and install the dependencies:

```bash
git clone <repository_url>
cd <repository_folder>
pip install -r requirements.txt
```

This will install all the necessary libraries for data processing, model training, and evaluation.

### Step 2: Download the ALEX-GYM-1 Dataset

The dataset for this project is available for download. You can download it from the provided link.

Once downloaded, extract the dataset and ensure the structure is as follows:

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

### Step 3: Organize the Code

The project directory is structured as follows:

```
<project_root>/
├── code/
│   ├── data/
│   ├── model_evaluation/
│   ├── model_training/
│   ├── utils/
│   ├── vision_model/
│   ├── pose_model/
│   ├── merged_model/
│   ├── train.py
│   └── eval.py
├── requirements.txt
└── ALEX-GYM-1/
```

- **data**: Contains scripts for data preprocessing, including the creation of training, validation, and test datasets.
- **model_evaluation**: Contains the code for evaluating models on test data.
- **model_training**: Includes code for training the models.
- **utils**: Contains utility scripts.
- **vision_model**: Contains the code for the vision-based model (3D CNN).
- **pose_model**: Contains the code for the pose-based model (joint angles and distances).
- **merged_model**: Contains the code for the combined (merged) model using both pose and vision features.
- **train.py**: The script to train a selected model (pose, vision, or merged).
- **eval.py**: The script to evaluate a trained model on the test dataset.

## 6. Running the Experiment

There are two main tasks you can perform: training the model or evaluating it on the test set.

### Option 1: Train the Model

To train a model, run the `train.py` script. You can choose between the **pose-based**, **vision-based**, or **merged** model. Here's how to execute the training process:

```bash
python code/train.py --base_path <path_to_dataset> --model_type <merged/pose/vision> --action_type <squat/deadlift/lunges> --epochs 100 --lr 1e-4 --batch_size 16 --device <cpu/cuda>
```

- `--base_path`: The path where your dataset is stored.
- `--model_type`: Choose between `merged`, `pose`, or `vision`.
- `--action_type`: Specify which exercise to train on (`squat`, `deadlift`, `lunges`).
- `--epochs`: Number of training epochs (e.g., 100).
- `--lr`: Learning rate for the optimizer (e.g., `1e-4`).
- `--batch_size`: Batch size for training (default is 16).
- `--device`: Specify whether to use CPU or CUDA for training.

Once executed, the training process will start, and the model will be trained for the selected exercise and model type.

### Option 2: Evaluate the Model

To evaluate a trained model, use the `eval.py` script. This will evaluate the performance of a saved model on the test set:

```bash
python code/eval.py --base_path <path_to_dataset> --model_type <merged/pose/vision> --action_type <squat/deadlift/lunges> --model_path <path_to_model>
```

- `--base_path`: The path where your dataset is stored.
- `--model_type`: Choose the model type (`merged`, `pose`, or `vision`).
- `--action_type`: Specify the exercise for evaluation (`squat`, `deadlift`, `lunges`).
- `--model_path`: Provide the path to the pre-trained model to evaluate.

The script will output the evaluation results, including the Hamming loss, which is used to assess the accuracy of the model's predictions for exercise classification.

### Output Example

After running either script, the output will include the training progress (for training) or evaluation metrics like Hamming loss (for evaluation).

### Step 4: Modify Configurations

You can easily modify the configurations of the training process (such as learning rate, batch size, etc.) by adjusting the corresponding arguments when running the scripts. This flexibility allows you to experiment with different settings to optimize model performance.

---
This work is submitted to the IJCNN 2025 conference. The authors are:

- **Ahmed Hassan**
- **Abdelaziz Essam**
- **Ahmed Yasser**

The project was conducted under the supervision of **Prof. Walid Gomaa** at the **Computer Science Engineering (CSE) Department**, **Egypt Japan University of Science and Technology (EJUST)**.



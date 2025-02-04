import torch.nn as nn
import torch
from ..vision_model.vision_model import DualInputResNet3D
from ..pose_model.pose_model import DualInputPose

class MultiModalModel(nn.Module):
    """
    Multi-modal model combining vision (ResNet3D) and pose (GRU-based) data 
    to perform prediction tasks based on features from both modalities.
    """
    def __init__(self, output_size=5, hidden_size = 512):
        super(MultiModalModel, self).__init__()
        try:
            self.vision_model = DualInputResNet3D(0,output_size=0)
            self.pose_model = DualInputPose(0,output_size=0)

            self.conv1d = nn.Sequential(
                nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

            self.fc = nn.Sequential(
                nn.Linear(64 * hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, output_size)
            )
        except Exception as e:
            print(f"Error in MultiModalModel initialization: {e}")

    def forward(self, image_frontal, image_lateral, pose_frontal, pose_lateral):
        """
        Forward pass of the MultiModalModel combining image and pose features.

        Args:
            image_frontal (Tensor): The frontal view video data.
            image_lateral (Tensor): The lateral view video data.
            pose_frontal (Tensor): The frontal pose data.
            pose_lateral (Tensor): The lateral pose data.
        
        Returns:
            Tensor: Final predicted output after processing both vision and pose data.
        """
        try:
            image_features = self.vision_model(image_frontal, image_lateral)
            pose_features = self.pose_model(pose_frontal, pose_lateral)

            combined_features = torch.stack((image_features, pose_features), dim=1)
            processed_features = self.conv1d(combined_features)

            flattened_features = processed_features.view(processed_features.size(0), -1)
            output = self.fc(flattened_features)
            return output
        except Exception as e:
            print(f"Error in MultiModalModel forward pass: {e}")
            return None
import torch
import torch.nn as nn
from .vision_model import DualInputResNet3D
from .pose_model import DualInputPose


class MultiModalModel(nn.Module):
    """
    Multi-modal model combining vision and pose streams with enhanced fusion mechanisms.
    """

    def __init__(self, output_size=6):
        super(MultiModalModel, self).__init__()
        try:
            # Both feature extractors now output 512-dimensional vectors
            self.vision_model = DualInputResNet3D(0, output_size=0)
            self.pose_model = DualInputPose(0, output_size=0)

            # Balanced fusion network with consistent dimensions
            self.fusion_layers = nn.Sequential(
                nn.Linear(512 + 512, 768),  # Combined 1024D → 768D
                nn.BatchNorm1d(768),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(768, 384),  # 768D → 384D
                nn.BatchNorm1d(384),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(384, 192),  # 384D → 192D
                nn.BatchNorm1d(192),
                nn.ReLU(),
                nn.Dropout(0.2),
            )

            # Classification head with progressive reduction
            self.classifier = nn.Sequential(
                nn.Linear(192, 96),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(96, output_size),
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

            combined_features = torch.cat((image_features, pose_features), dim=1)
            fused_features = self.fusion_layers(combined_features)

            output = self.classifier(fused_features)
            return output
        except Exception as e:
            print(f"Error in MultiModalModel forward pass: {e}")
            return None

import torch
import torch.nn as nn
import torchvision.models.video as models


class PretrainedResNet3D(nn.Module):
    """
    A wrapper around the 3D ResNet model (ResNet3D) to leverage pretrained weights and provide feature extraction.
    The final fully connected layer is replaced with an identity layer, making it suitable for downstream tasks
    like feature extraction for multi-input models.

    Args:
        pretrained (bool, optional): Whether to load the pretrained weights for ResNet3D (default is True).

    Attributes:
        resnet3d (nn.Module): The ResNet3D model with the final layer replaced with an identity layer.
    """

    def __init__(self, pretrained=True):
        super(PretrainedResNet3D, self).__init__()
        try:
            # Load the pretrained ResNet3D model
            self.resnet3d = models.r3d_18(pretrained=pretrained)
            # Replace the final fully connected layer with an identity layer
            self.resnet3d.fc = nn.Identity()

            # Freeze early layers to prevent overfitting
            for name, param in self.resnet3d.named_parameters():
                if "layer3" not in name and "layer4" not in name:
                    param.requires_grad = False
        except Exception as e:
            print(f"Error in PretrainedResNet3D initialization: {e}")

    def forward(self, x):
        """
        Forward pass through the ResNet3D model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, num_frames, height, width].

        Returns:
            torch.Tensor: Extracted feature vector after passing through the ResNet3D model.
        """
        try:
            return self.resnet3d(x)
        except Exception as e:
            print(f"Error in PretrainedResNet3D forward pass: {e}")
            return None


class DualInputResNet3D(nn.Module):
    """
    A model that uses two separate streams of the 3D ResNet architecture with enhanced fusion.
    """

    def __init__(self, MergedOrAlone, output_size=6, hidden_size=512, dropout_rate=0.3):
        super(DualInputResNet3D, self).__init__()
        try:
            # Pretrained ResNet3D streams for frontal and lateral views
            self.resnet3d_frontal = PretrainedResNet3D()
            self.resnet3d_lateral = PretrainedResNet3D()

            combined_dim = hidden_size * 2

            # Path selection logic
            if MergedOrAlone == 1:
                # Standalone prediction path with advanced normalization
                self.fc_layers = nn.Sequential(
                    nn.Linear(combined_dim, combined_dim // 2),  # 1024 → 512
                    nn.BatchNorm1d(combined_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(combined_dim // 2, combined_dim // 4),  # 512 → 256
                    nn.ReLU(),
                    nn.Dropout(dropout_rate * 0.7),  # Reduced dropout
                    nn.Linear(combined_dim // 4, output_size),  # Final prediction
                )
            else:
                # Maintain dimension for fusion - consistent with pose stream
                self.fc_layers = nn.Sequential(
                    nn.Linear(combined_dim, hidden_size),  # 1024 → 512
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate * 0.7),
                )
        except Exception as e:
            print(f"Error in DualInputResNet3D initialization: {e}")

    def forward(self, frontal, lateral):
        """
        Forward pass through the dual input ResNet3D model.

        Args:
            frontal (torch.Tensor): Input tensor for the frontal view, shape [batch_size, channels, num_frames, height, width].
            lateral (torch.Tensor): Input tensor for the lateral view, shape [batch_size, channels, num_frames, height, width].

        Returns:
            torch.Tensor: Final output tensor, shape [batch_size, output_size] or feature tensor for fusion.
        """
        try:
            # Permute to [batch_size, channels, num_frames, height, width]
            frontal = frontal.permute(0, 2, 1, 3, 4).contiguous()
            lateral = lateral.permute(0, 2, 1, 3, 4).contiguous()

            # Process frontal input through ResNet3D
            x_f = self.resnet3d_frontal(frontal)
            x_f = x_f.view(x_f.size(0), -1)  # Flatten features

            # Process lateral input through ResNet3D
            x_l = self.resnet3d_lateral(lateral)
            x_l = x_l.view(x_l.size(0), -1)  # Flatten features

            # Concatenate features from frontal and lateral streams
            x = torch.cat((x_f, x_l), dim=1)

            # Pass through fully connected layers
            x = self.fc_layers(x)

            return x
        except Exception as e:
            print(f"Error in DualInputResNet3D forward pass: {e}")
            return None

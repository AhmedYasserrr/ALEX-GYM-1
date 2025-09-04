import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    An optimized residual block with balanced main and shortcut paths for improved gradient flow.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        try:
            # Main path with bottleneck architecture for efficiency
            self.main_path = nn.Sequential(
                # Reduce channels first (bottleneck)
                nn.Conv2d(in_channels, out_channels // 2, kernel_size=(1, 1), stride=1),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(),
                # Extract features with larger kernel
                nn.Conv2d(
                    out_channels // 2,
                    out_channels // 2,
                    kernel_size=(3, 1),
                    stride=(stride, 1),
                    padding=(1, 0),
                ),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(),
                # Expand channels back
                nn.Conv2d(
                    out_channels // 2, out_channels, kernel_size=(1, 1), stride=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.2),  # Lower dropout for better feature retention
            )

            # Adaptive shortcut connection
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=(1, 1),
                        stride=(stride, 1),
                    ),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.shortcut = nn.Identity()

        except Exception as e:
            print(f"Error in ResidualBlock initialization: {e}")

    def forward(self, x):
        """
        Forward pass of the enhanced ResidualBlock with pre-activation design.
        """
        try:
            # Process paths
            residual = self.shortcut(x)
            main_output = self.main_path(x)

            # Combine and activate
            x = main_output + residual
            return F.relu(x)  # Apply ReLU after addition
        except Exception as e:
            print(f"Error in ResidualBlock forward pass: {e}")
            return None


class Pose_Model(nn.Module):
    """
    A CNN model for extracting features from pose data with enhanced architecture.
    """

    def __init__(
        self, input_channels=5984, hidden_dim=256, expansion_factor=2, feature_dim=512
    ):
        super(Pose_Model, self).__init__()
        try:
            mid_channels = hidden_dim * expansion_factor

            # Initial dimensionality reduction
            self.initial_conv = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    hidden_dim,
                    kernel_size=(3, 1),
                    stride=(2, 1),
                    padding=(1, 0),
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            )

            # Enhanced multi-stage residual pathway
            self.res_stage1 = ResidualBlock(hidden_dim, mid_channels, stride=2)
            self.res_stage2 = ResidualBlock(mid_channels, feature_dim, stride=1)

            # Spatial pooling to further compact the representation
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        except Exception as e:
            print(f"Error in Pose_Model initialization: {e}")

    def forward(self, x):
        """
        Forward pass with enhanced feature extraction.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_frames, channels]

        Returns:
            torch.Tensor: Features of shape [batch_size, feature_dim]
        """
        try:
            batch_size, num_frames, channels = x.shape
            x = x.permute(0, 2, 1).unsqueeze(
                -1
            )  # [batch_size, channels, num_frames, 1]

            # Multi-stage feature extraction
            x = self.initial_conv(x)
            x = self.res_stage1(x)
            x = self.res_stage2(x)

            # Global pooling and flatten
            x = self.pool(x)
            x = x.view(batch_size, -1)  # [batch_size, feature_dim]

            return x

        except Exception as e:
            print(f"Error in Pose_Model forward pass: {e}")
            return None


class DualInputPose(nn.Module):
    """
    A model that processes pose data from two views with enhanced feature extraction.
    """

    def __init__(
        self, MergedOrAlone=1, output_size=6, pose_input_channels=5984, feature_dim=512
    ):
        super(DualInputPose, self).__init__()
        try:
            # Pose models for front and lateral views with consistent output dimension
            self.front_model = Pose_Model(
                input_channels=pose_input_channels, feature_dim=feature_dim
            )
            self.lat_model = Pose_Model(
                input_channels=pose_input_channels, feature_dim=feature_dim
            )

            combined_dim = feature_dim * 2

            # Path selection based on standalone vs. merged operation
            if MergedOrAlone == 1:
                # Progressive reduction for standalone predictions
                self.fc = nn.Sequential(
                    nn.Linear(combined_dim, combined_dim // 2),  # 1024 → 512
                    nn.BatchNorm1d(combined_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(combined_dim // 2, combined_dim // 4),  # 512 → 256
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(combined_dim // 4, output_size),  # 256 → output
                )
            else:
                # Maintain dimension for fusion - output same size as individual streams for balanced fusion
                self.fc = nn.Sequential(
                    nn.Linear(combined_dim, feature_dim),  # 1024 → 512
                    nn.BatchNorm1d(feature_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                )
        except Exception as e:
            print(f"Error in DualInputPose initialization: {e}")

    def forward(self, front_input, lat_input):
        """
        Forward pass of the DualInputPose model.

        Args:
            front_input (torch.Tensor): Input tensor for frontal pose data.
            lat_input (torch.Tensor): Input tensor for lateral pose data.

        Returns:
            torch.Tensor: The predicted output or feature tensor for fusion.
        """
        try:
            # Pass through the front and lateral Pose models
            front_features = self.front_model(
                front_input
            )  # Shape: [batch_size, feature_dim]
            lat_features = self.lat_model(lat_input)  # Shape: [batch_size, feature_dim]

            # Concatenate features
            combined_features = torch.cat(
                (front_features, lat_features), dim=1
            )  # Shape: [batch_size, feature_dim * 2]

            # Process through FC layers
            output = self.fc(combined_features)

            return output
        except Exception as e:
            print(f"Error in DualInputPose forward pass: {e}")
            return None

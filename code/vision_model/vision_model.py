import torch.nn as nn
import torch
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
    
    Methods:
        forward(x): Performs a forward pass through the ResNet3D model, returning the extracted features.

    """
    def __init__(self, pretrained=True):
        super(PretrainedResNet3D, self).__init__()
        try:
            # Load the pretrained ResNet3D model
            self.resnet3d = models.r3d_18(pretrained=pretrained)
            # Replace the final fully connected layer with an identity layer
            self.resnet3d.fc = nn.Identity()

            for name, param in self.resnet3d.named_parameters():
                if "layer3" not in name and "layer4" not in name:
                    param.requires_grad = False
        except Exception as e:
            print(f"Error in PretrainedResNet3D initialization: {e}")

    def forward(self, x):
        """
        Forward pass through the ResNet3D model. This method processes the input tensor through the ResNet3D 
        architecture, extracting features by passing through the layers.

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
    A model that uses two separate streams of the 3D ResNet architecture for frontal and lateral inputs.
    The features extracted from both streams are concatenated and passed through fully connected layers
    to predict the final output (e.g., exercise classification, pose evaluation).

    Args:
        output_size (int, optional): The number of output classes or features (default is 6).
        feature_dim (int, optional): The feature dimension after flattening the ResNet3D output (default is 512).
        dropout_rate (float, optional): Dropout rate for regularization (default is 0.3).
    
    Attributes:
        resnet3d_frontal (PretrainedResNet3D): Pretrained ResNet3D model for the frontal view input.
        resnet3d_lateral (PretrainedResNet3D): Pretrained ResNet3D model for the lateral view input.
        fc_layers (nn.Sequential): Fully connected layers to process concatenated features from both streams.

    Methods:
        forward(frontal, lateral): Performs a forward pass through the dual ResNet3D streams and fully connected layers.

    """
    def __init__(self,MergedOrAlone,output_size, hidden_size=512, dropout_rate=0.3):
        super(DualInputResNet3D, self).__init__()
        try:
            # Pretrained ResNet3D streams for frontal and lateral inputs
            self.resnet3d_frontal = PretrainedResNet3D()
            self.resnet3d_lateral = PretrainedResNet3D()
            # if mergedOrAlone == 1, it will work as standalone model , else: it will work in the merged model
            if MergedOrAlone == 1:
                # Fully connected layers using nn.Sequential
                self.fc_layers = nn.Sequential(
                    nn.Linear(hidden_size * 2, 1024),  # Combine features from both inputs
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(1024, hidden_size),  # Intermediate layer
                    nn.ReLU(),
                    nn.Linear(hidden_size, output_size)  # Final output layer 
                )
            else:
                # Fully connected layers using nn.Sequential
                self.fc_layers = nn.Sequential(
                    nn.Linear(hidden_size * 2, 1024),  # Combine features from both inputs
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(1024, hidden_size),  # Intermediate layer
                )
        except Exception as e:
            print(f"Error in DualInputResNet3D initialization: {e}")

    def forward(self, frontal, lateral):
        """
        Forward pass through the dual input ResNet3D model. The inputs (frontal and lateral views) are processed
        through separate ResNet3D models, and the extracted features are concatenated and passed through fully
        connected layers to produce the final output.

        Args:
            frontal (torch.Tensor): Input tensor for the frontal view, shape [batch_size, channels, num_frames, height, width].
            lateral (torch.Tensor): Input tensor for the lateral view, shape [batch_size, channels, num_frames, height, width].

        Returns:
            torch.Tensor: Final output tensor, shape [batch_size, output_size], where output_size is the number of classes or features.
        """
        try:
            # Permute to [batch_size, channels, num_frames, height, width]
            frontal = frontal.permute(0, 2, 1, 3, 4).contiguous()
            lateral = lateral.permute(0, 2, 1, 3, 4).contiguous()

            # Process frontal input through ResNet3D
            x_f = self.resnet3d_frontal(frontal)
            x_f = x_f.view(x_f.size(0), -1)  # Flatten features
            print(f"Frontal features shape: {x_f.shape}")  # Debugging

            # Process lateral input through ResNet3D
            x_l = self.resnet3d_lateral(lateral)
            x_l = x_l.view(x_l.size(0), -1)  # Flatten features
            print(f"Lateral features shape: {x_l.shape}")  # Debugging

            # Concatenate features from frontal and lateral streams
            x = torch.cat((x_f, x_l), dim=1)
            print(f"Concatenated features shape: {x.shape}")  # Debugging

            # Pass through fully connected layers
            x = self.fc_layers(x)
            print(f"Final output shape: {x.shape}")  # Debugging

            return x
        except Exception as e:
            print(f"Error in DualInputResNet3D forward pass: {e}")
            return None

import torch.nn as nn
import torch

class ResidualBlock(nn.Module):
    """
    A residual block with three convolutional layers and a shortcut connection.
    The block applies a series of convolutions with batch normalization and ReLU activations,
    then adds the input (shortcut) to the output (residual connection) to help alleviate the vanishing gradient problem.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): The stride for the first convolution (default is 1). This can be used to reduce spatial dimensions.
    
    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        bn1 (nn.BatchNorm2d): The batch normalization layer after the first convolution.
        relu (nn.ReLU): The activation function after each convolution.
        conv2 (nn.Conv2d): The second convolutional layer.
        bn2 (nn.BatchNorm2d): The batch normalization layer after the second convolution.
        conv3 (nn.Conv2d): The third convolutional layer.
        bn3 (nn.BatchNorm2d): The batch normalization layer after the third convolution.
        shortcut (nn.Conv2d): The shortcut connection to match the output dimensions of the residual block.
        shortcut_bn (nn.BatchNorm2d): The batch normalization layer for the shortcut connection.

    Methods:
        forward(x): Performs a forward pass through the residual block.
        
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        try:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0))
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()

            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0))
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), stride=1, padding=(1, 0))
            self.bn3 = nn.BatchNorm2d(out_channels)

            # Shortcut connection
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, 1))
            self.shortcut_bn = nn.BatchNorm2d(out_channels)
        except Exception as e:
            print(f"Error in ResidualBlock initialization: {e}")

    def forward(self, x):
        """
        Forward pass of the ResidualBlock. The input is passed through three convolutional layers with batch 
        normalization and ReLU activations, and the input is added to the output through a shortcut connection.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, height, width] after applying the residual block.
        """
        try:
            shortcut = self.shortcut(x)
            shortcut = self.shortcut_bn(shortcut)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)

            x = x + shortcut
            x = self.relu(x)
            return x
        except Exception as e:
            print(f"Error in ResidualBlock forward pass: {e}")
            return None


class Pose_Model(nn.Module):
    """
    A convolutional neural network (CNN) model for extracting temporal features from pose data.

    This model takes pose data as input (in the form of a tensor with shape [batch_size, num_frames, channels]) and processes
    it through a series of 2D convolutional layers followed by residual blocks, and then aggregates temporal features
    using a GRU (Gated Recurrent Unit).

    Args:
        input_channels (int, optional): Number of input channels (default is 5984). 
                                         This is the length of the feature vector for each frame.
        residual_channels1 (int, optional): Number of channels for the first residual block (default is 256).
        residual_channels2 (int, optional): Number of channels for the second residual block (default is 128).
        residual_channels3 (int, optional): Number of channels for the third residual block (default is 64).
        final_channels (int, optional): Number of channels for the final convolution (default is 32).
        gru_hidden_size (int, optional): Hidden size of the GRU layer (default is 128).

    Attributes:
        initial_conv (nn.Sequential): The first convolutional layer.
        residual_block1 (ResidualBlock): First residual block for feature refinement.
        residual_block2 (ResidualBlock): Second residual block for feature refinement.
        final_conv (nn.Sequential): Final convolutional layer to reduce the number of channels.
        gru (nn.GRU): GRU layer to aggregate temporal features across frames.

    Methods:
        forward(x): Passes input data through the model and returns temporal feature representation.

    """
    def __init__(self, input_channels=5984, residual_channels1=256, residual_channels2=128, residual_channels3=64, final_channels=32, gru_hidden_size=128):
        super(Pose_Model, self).__init__()
        try:
            # Initial 2D Convolution
            self.initial_conv = nn.Sequential(
                nn.Conv2d(input_channels, residual_channels1, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
                nn.BatchNorm2d(residual_channels1),
                nn.ReLU()
            )

            # Two Residual Blocks
            self.residual_block1 = ResidualBlock(residual_channels1, residual_channels2, stride=2)
            self.residual_block2 = ResidualBlock(residual_channels2, residual_channels3, stride=2)

            # Final 2D Convolution to reduce channels
            self.final_conv = nn.Sequential(
                nn.Conv2d(residual_channels3, final_channels, kernel_size=(1, 1)),
                nn.BatchNorm2d(final_channels),
                nn.ReLU()
            )

            # GRU for temporal aggregation
            self.gru = nn.GRU(input_size=final_channels, hidden_size=gru_hidden_size, batch_first=True)
        except Exception as e:
            print(f"Error in Pose_Model initialization: {e}")

    def forward(self, x):
        """
        Forward pass of the model. The input is processed through the convolutional layers, followed by residual blocks, 
        and temporal features are extracted using the GRU.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_frames, channels] containing pose data.
        
        Returns:
            torch.Tensor: Temporal features of shape [batch_size, gru_hidden_size] extracted by the GRU.
        """
        try:
            batch_size, num_frames, channels = x.shape
            x = x.permute(0, 2, 1).unsqueeze(-1)  # Reshape to [batch_size, input_channels, num_frames, 1]

            # Pass through the convolutional layers
            x = self.initial_conv(x)
            x = self.residual_block1(x)
            x = self.residual_block2(x)

            x = self.final_conv(x)  # Shape: [batch_size, final_channels, num_frames//4, 1]

            # Prepare for GRU
            x = x.squeeze(-1).permute(0, 2, 1)  # Reshape to [batch_size, num_frames//4, final_channels]

            # GRU for temporal features
            _, pooled_features = self.gru(x)  # Shape: [1, batch_size, gru_hidden_size]
            return pooled_features.squeeze(0)  # Shape: [batch_size, gru_hidden_size]
        except Exception as e:
            print(f"Error in Pose_Model forward pass: {e}")
            return None


class DualInputPose(nn.Module):
    """
    A model that processes pose data from two views: frontal and lateral. It combines the features from both views
    using two separate Pose_Model instances and then makes predictions using a fully connected layer.

    Args:
        pose_input_channels (int, optional): Number of input channels (default is 5984).
        hidden_size (int, optional): Hidden size for the GRU in the Pose_Model (default is 128).
        output_size (int, optional): The number of output features from the final fully connected layers (default is FeatureNum).

    Attributes:
        front_model (Pose_Model): The Pose_Model instance for processing frontal pose data.
        lat_model (Pose_Model): The Pose_Model instance for processing lateral pose data.
        fc (nn.Sequential): A series of fully connected layers for prediction based on combined features.

    Methods:
        forward(front_input, lat_input): Passes both the front and lateral input data through respective PoseModels 
                                         and returns the predicted output.
        
    Example:

    """
    def __init__(self,MergedOrAlone,output_size,pose_input_channels=5984, hidden_size=128):
        super(DualInputPose, self).__init__()
        try:
            # Pose models for front and lateral views
            self.front_model = Pose_Model(input_channels=pose_input_channels, gru_hidden_size=hidden_size)
            self.lat_model = Pose_Model(input_channels=pose_input_channels, gru_hidden_size=hidden_size)
            
            # if mergedOrAlone == 1, it will work as standalone model , else: it will work in the merged model
            if MergedOrAlone == 1:
                # Fully connected layers for prediction
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size * 2, 512),
                    nn.Linear(512,128),
                    nn.Linear(128,output_size)
                )
            else:
                # Fully connected layers for prediction
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size * 2, 512),
                )
        except Exception as e:
            print(f"Error in DualInputPose initialization: {e}")

    def forward(self, front_input, lat_input):
        """
        Forward pass of the DualInputPose model. The model processes the front and lateral views of pose data 
        through separate Pose_Model instances, concatenates the resulting features, and makes predictions.

        Args:
            front_input (torch.Tensor): Input tensor of shape [batch_size, num_frames, channels] for frontal pose data.
            lat_input (torch.Tensor): Input tensor of shape [batch_size, num_frames, channels] for lateral pose data.
        
        Returns:
            torch.Tensor: The predicted output of shape [batch_size, output_size].
        """
        try:
            # Pass through the front and lateral Pose models
            front_features = self.front_model(front_input)  # Shape: [batch_size, hidden_size]
            print(f"Front pose features shape: {front_features.shape}")  # Debugging

            lat_features = self.lat_model(lat_input)  # Shape: [batch_size, hidden_size]
            print(f"Lateral pose features shape: {lat_features.shape}")  # Debugging

            # Concatenate features
            combined_features = torch.cat((front_features, lat_features), dim=1)  # Shape: [batch_size, hidden_size * 2]
            print(f"Combined pose features shape: {combined_features.shape}")  # Debugging

            # Predict criteria
            output = self.fc(combined_features)  # Shape: [batch_size, output_size]
            print(f"Final pose features shape: {output.shape}")  # Debugging

            return output
        except Exception as e:
            print(f"Error in DualInputPose forward pass: {e}")
            return None

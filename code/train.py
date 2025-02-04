import argparse
import os
from data.process_data import process_action_data
from data.image_preprocessing import preprocess_and_save_images
from data.data_loader import create_dataloader
from merged_model.merged_model import MultiModalModel
from pose_model.pose_model import DualInputPose
from vision_model.vision_model import DualInputResNet3D
from training.model_training import train_combined_model

def main():
    """
    Main function to train a model for exercise evaluation and classification.
    
    This function parses command-line arguments, preprocesses the input data,
    sets up dataloaders, initializes the model based on the user-selected options,
    and trains the model using the specified parameters.

    Arguments:
        None: All required arguments are passed via command-line arguments.
    """

    # Set up argparse to accept command-line arguments
    parser = argparse.ArgumentParser(description="Train a model for exercise evaluation and classification.")

    # Argument for the base path where the dataset is located
    parser.add_argument('--base_path', type=str, required=True,
                        help='The base path where the dataset is located')

    # Argument for the type of model to use ('merged', 'pose', 'vision')
    parser.add_argument('--model_type', type=str, choices=['merged', 'pose', 'vision'], default='merged',
                        help='The type of model to use: "merged", "pose", or "vision"')

    # Argument for the type of action to train on ('squat', 'deadlift', 'lunges')
    parser.add_argument('--action_type', type=str, choices=['squat', 'deadlift', 'lunges'], default='squat',
                        help='The action type to train on: "squat", "deadlift", or "lunges"')

    # Argument for the number of epochs for training
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')

    # Argument for the learning rate for the optimizer
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')

    # Argument for the batch size for training
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')

    # Argument to specify the device for training ('cpu' or 'cuda')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use for training')

    # Argument for the gradient clipping norm
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm')

    # Argument for the number of epochs with no improvement for early stopping
    parser.add_argument('--patience', type=int, default=10, help='Number of epochs with no improvement for early stopping')
    
    # Parse the arguments
    args = parser.parse_args()

    # Set the feature number based on the chosen action type (squat, deadlift, or lunges)
    feature_number = 0
    if args.action_type == 'squat':
        feature_number = 6
    elif args.action_type == 'deadlift':
        feature_number = 5
    else:
        feature_number = 7

    # Define directories for raw and preprocessed image data
    input_dir = os.path.join(args.base_path, f"{args.action_type}_Frames")  # Directory containing raw images
    output_dir = os.path.join(args.base_path, 'preprocessed_images', args.action_type)  # Directory to save preprocessed images

    # Preprocess and save images to the output directory
    preprocess_and_save_images(input_dir, output_dir)

    # Process action data (e.g., labels, metadata) and split into train, validation, and test sets
    train_df, val_df, test_df = process_action_data(args.base_path, args.action_type)

    # Create data loaders for training and evaluation datasets
    train_dataloader = create_dataloader(train_df, batch_size=args.batch_size)
    eval_dataloader = create_dataloader(val_df, batch_size=args.batch_size)
    rating_model = 0
    # Initialize the model based on the selected model type
    if args.model_type == 'merged':
        rating_model = MultiModalModel(feature_number)  # Merged model
    elif args.model_type == 'pose':
        rating_model = DualInputPose(1, feature_number)  # Pose model
    else:
        rating_model = DualInputResNet3D(1, feature_number)  # Vision model

    # Train the selected model using the defined parameters
    train_losses, hamming_losses = train_combined_model(
        CustomModel=rating_model,
        model_type=args.model_type,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        clip_grad_norm=args.clip_grad_norm,
        patience=args.patience
    )

    # Output training results
    print("Training complete.")
    print("Train losses:", train_losses)
    print("Hamming losses:", hamming_losses)


if __name__ == '__main__':
    # Run the main function
    main()

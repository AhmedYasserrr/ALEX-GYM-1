import torch
import torch.nn as nn

def evaluate_combined_model(CustomModel,model_type, dataloader, device):
    """
    Evaluates the model and computes the Hamming loss for each feature.

    Args:
        CustomModel (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader providing the evaluation data.
        loss_fn (nn.Module): The loss function for rating predictions.
        device (str): Device to run the model ('cpu' or 'cuda').

    Returns:
        float: Average evaluation loss.
        dict: Hamming distance for each feature.
    """
    CustomModel.eval()  # Set model to evaluation mode
    total_loss = 0.0
    num_features = len(dataloader.dataset.dataset.ratings)  # Get number of features 
    
    # Initialize dictionaries to track TP, TN, FP, FN counts for each feature
    metrics = {feature_idx: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} for feature_idx in range(num_features)}
    total_samples = {feature_idx: 0 for feature_idx in range(num_features)}
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move batch data to the device
            image_frontal, image_lateral, pose_frontal, pose_lateral, label_class, ratings = [tensor.to(device) for tensor in batch]

            if model_type == 'merged':
                # Forward pass through the model
                ratings_output = CustomModel(image_frontal, image_lateral, pose_frontal, pose_lateral)
            elif model_type == 'vision':
                # Forward pass through the model
                ratings_output = CustomModel(image_frontal, image_lateral)
            else:
                # Forward pass through the model
                ratings_output = CustomModel(pose_frontal, pose_lateral)

            # Compute loss between predicted ratings and actual ratings
            total_loss += loss_fn(ratings_output, ratings)

            # Sigmoid activation for binary predictions
            predicted_ratings = torch.sigmoid(ratings_output) > 0.5
            actual_ratings = ratings.byte()

            # Compute TP, TN, FP, FN for each feature
            for feature_idx in range(num_features):
                preds = predicted_ratings[:, feature_idx]
                trues = actual_ratings[:, feature_idx]

                metrics[feature_idx]['TP'] += (preds & trues).sum().item()
                metrics[feature_idx]['TN'] += (~preds & ~trues).sum().item()
                metrics[feature_idx]['FP'] += (preds & ~trues).sum().item()
                metrics[feature_idx]['FN'] += (~preds & trues).sum().item()
                total_samples[feature_idx] += trues.numel()

    # Calculate Hamming distance for each feature
    hamming_distances = {feature_idx: (metrics[feature_idx]['FP'] + metrics[feature_idx]['FN']) / total_samples[feature_idx]
                         if total_samples[feature_idx] > 0 else 0.0
                         for feature_idx in range(num_features)}

    avg_loss = total_loss / len(dataloader)
    # Calculate the mean Hamming distance across all features
    mean_hamming_loss = sum(hamming_distances.values()) / len(hamming_distances)

    return avg_loss, mean_hamming_loss
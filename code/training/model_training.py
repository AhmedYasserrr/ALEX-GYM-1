import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from ..evaluation.model_evaluation  import evaluate_combined_model

def train_combined_model(CustomModel, model_type,train_dataloader, eval_dataloader, epochs=1000, lr=1e-4,
                         device='cpu', clip_grad_norm=1.0, patience=10):
    
    """
    Trains a combined model for exercise evaluation and classification, using a binary cross-entropy loss for rating prediction and 
    Hamming loss for evaluation. Implements early stopping based on validation loss, learning rate adjustment, and model checkpointing.

    Args:
        CustomModel (torch.nn.Module): The model to be trained, typically a neural network that predicts exercise ratings.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        eval_dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation/validation dataset.
        epochs (int, optional): The number of epochs to train the model for. Default is 1000.
        lr (float, optional): Learning rate for the optimizer. Default is 1e-4.
        device (str, optional): The device to train the model on ('cpu' or 'cuda'). Default is 'cpu'.
        clip_grad_norm (float, optional): The maximum gradient norm for gradient clipping. Default is 1.0.
        patience (int, optional): The number of epochs with no improvement in validation loss before early stopping is triggered. Default is 10.

    Returns:
        tuple: A tuple containing two lists:
            - train_losses (list): A list of average training losses for each epoch.
            - hamming_losses (list): A list of mean Hamming losses for each epoch during evaluation.
    """
    # Move the model to the specified device (CPU or GPU)
    CustomModel.to(device)

    # Set up optimizer and loss function for training
    optimizer = optim.Adam(list(CustomModel.parameters()), lr=lr)
    hamming_loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for rating task

    # Learning rate scheduler to adjust learning rate if validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Track best evaluation loss and patience for early stopping
    best_eval_loss = float('inf')
    best_mean_hamming_loss = float('inf')
    patience_counter = 0
    
    # Lists to track training loss and validation Hamming loss for plotting
    train_losses = []
    hamming_losses = []

    print("Training the model...")

    # Training loop over epochs
    for epoch in range(epochs):
        CustomModel.train()  # Set model to training mode
        running_train_loss = 0.0

        # Iterate over batches in the training data
        for batch_idx, batch in enumerate(train_dataloader):
            # Unpack the batch and move to the specified device
            image_frontal, image_lateral, pose_frontal, pose_lateral, label_class, ratings = [tensor.to(device) for tensor in batch]
            
            optimizer.zero_grad()  # Zero out the gradients from the previous step
            
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
            train_loss = hamming_loss_fn(ratings_output, ratings)

            # Backward pass and optimization
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(CustomModel.parameters()), clip_grad_norm)
            optimizer.step()

            running_train_loss += train_loss.item()

            # Print the training progress for the current batch
            print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_dataloader)}], Train Loss: {train_loss.item():.4f}")

        # Calculate average training loss for the epoch
        avg_train_loss = running_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)  # Append train loss to list for plotting

        # Evaluate the model on the validation data after each epoch
        eval_loss, mean_hamming_loss = evaluate_combined_model(CustomModel, eval_dataloader, device)
        
        # Log validation Hamming loss for plotting
        hamming_losses.append(mean_hamming_loss)

        print(f"Epoch [{epoch + 1}/{epochs}] Summary: Train Loss: {avg_train_loss:.4f}, Eval Loss: {eval_loss:.4f}, Mean Hamming Loss: {mean_hamming_loss:.4f}")

        # Scheduler step based on the evaluation loss
        scheduler.step(eval_loss)

        # Early stopping based on validation loss
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            if mean_hamming_loss < best_mean_hamming_loss:
                best_mean_hamming_loss = mean_hamming_loss
                print(f"Best Hamming Loss: {best_mean_hamming_loss}")
            patience_counter = 0
            # Save the best model based on validation loss
            torch.save(CustomModel.state_dict(), f"best_eval_model_epoch_{epoch + 1}.pt")
            print("Model checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    epochs = range(1, len(train_losses) + 1)

    # Plot Training Loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='b')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # Plot Hamming Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hamming_losses, label='Validation Hamming Loss', color='r')
    plt.title("Validation Hamming Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Hamming Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    print("Training complete.")

    return train_losses, hamming_losses
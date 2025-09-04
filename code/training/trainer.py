import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..utils.evaluation_metrics import calculate_metrics


class Trainer:
    """
    Handles the training, validation, and testing of models.

    This class abstracts away the training loop and evaluation procedures,
    supporting different model types (vision, pose, multimodal).
    """

    def __init__(self, model, config, train_loader, val_loader, test_loader=None):
        """
        Initialize the trainer with model, data, and configuration.

        Args:
            model: The PyTorch model to train
            config: TrainingConfig object with training parameters
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: Optional DataLoader for test data
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = config.device

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", patience=5, factor=0.5
        )

        # Tracking variables
        self.best_val_loss = float("inf")
        self.best_hamming_distance = float("inf")
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "hamming_distance": [],
            "f1_score": [],
        }

    def train(self):
        """Run the complete training process."""
        print(f"Starting training on {self.device}...")
        print(f"Model type: {self.config.model_type}")

        start_time = time.time()

        for epoch in range(self.config.epochs):
            # Training phase
            train_loss = self._train_epoch(epoch)
            self.history["train_loss"].append(train_loss)

            # Validation phase
            val_loss, metrics = self._validate_epoch(epoch)
            self.history["val_loss"].append(val_loss)
            self.history["hamming_distance"].append(metrics["hamming_distance"])
            self.history["f1_score"].append(metrics["f1_score"])

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_model(f"best_model_epoch_{epoch+1}_loss.pt")
                print("Model saved based on validation loss.")
            elif metrics["hamming_distance"] < self.best_hamming_distance:
                self.best_hamming_distance = metrics["hamming_distance"]
                self.patience_counter = 0
                self._save_model(f"best_model_epoch_{epoch+1}_hamming.pt")
                print("Model saved based on Hamming distance.")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

        # Final model save
        self._save_model("final_model.pt")

        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time/60:.2f} minutes")

        # Plot training history
        self._plot_training_history()

        return self.history

    def _train_epoch(self, epoch):
        """
        Train the model for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            float: Average training loss for this epoch
        """
        self.model.train()
        running_loss = 0.0

        # Progress bar for batches
        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch+1}/{self.config.epochs}",
        )

        for batch_idx, batch in progress_bar:
            # Extract data based on model type
            if self.config.model_type == "pose":
                front_pose, lat_pose, _, labels = self._extract_pose_data(batch)
                outputs = self.model(front_pose, lat_pose)

            elif self.config.model_type == "vision":
                front_frames, lat_frames, _, labels = self._extract_vision_data(batch)
                outputs = self.model(front_frames, lat_frames)

            elif self.config.model_type == "multimodal":
                front_frames, lat_frames, front_pose, lat_pose, labels = (
                    self._extract_multimodal_data(batch)
                )
                outputs = self.model(front_frames, lat_frames, front_pose, lat_pose)

            # Compute loss
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.clip_grad_norm
                )

            self.optimizer.step()

            # Update running loss
            running_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{running_loss/(batch_idx+1):.4f}",
                }
            )

        avg_loss = running_loss / len(self.train_loader)
        print(f"Epoch {epoch+1} - Training Loss: {avg_loss:.4f}")

        return avg_loss

    def _validate_epoch(self, epoch):
        """
        Validate the model on the validation set.

        Args:
            epoch: Current epoch number

        Returns:
            tuple: (validation_loss, metrics_dict)
        """
        self.model.eval()
        val_loss = 0.0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Extract data based on model type
                if self.config.model_type == "pose":
                    front_pose, lat_pose, _, labels = self._extract_pose_data(batch)
                    outputs = self.model(front_pose, lat_pose)

                elif self.config.model_type == "vision":
                    front_frames, lat_frames, _, labels = self._extract_vision_data(
                        batch
                    )
                    outputs = self.model(front_frames, lat_frames)

                elif self.config.model_type == "multimodal":
                    front_frames, lat_frames, front_pose, lat_pose, labels = (
                        self._extract_multimodal_data(batch)
                    )
                    outputs = self.model(front_frames, lat_frames, front_pose, lat_pose)

                # Compute loss
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                # Store predictions and targets for metrics calculation
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        # Combine batch results
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)

        # Calculate metrics
        metrics = calculate_metrics(all_predictions, all_targets)

        # Calculate average validation loss
        avg_val_loss = val_loss / len(self.val_loader)

        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")
        print(
            f"Metrics: Hamming Loss = {metrics['hamming_distance']:.4f}, F1 Score = {metrics['f1_score']:.4f}"
        )
        print(
            f"Precision = {metrics['precision']:.4f}, Recall = {metrics['recall']:.4f}"
        )

        return avg_val_loss, metrics

    def test(self, model_path=None):
        """
        Test the model on the test dataset.

        Args:
            model_path: Optional path to a saved model to test

        Returns:
            dict: Test metrics
        """
        if self.test_loader is None:
            print("No test data loader provided.")
            return None

        if model_path:
            self._load_model(model_path)

        self.model.eval()
        test_loss = 0.0

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                # Extract data based on model type
                if self.config.model_type == "pose":
                    front_pose, lat_pose, _, labels = self._extract_pose_data(batch)
                    outputs = self.model(front_pose, lat_pose)

                elif self.config.model_type == "vision":
                    front_frames, lat_frames, _, labels = self._extract_vision_data(
                        batch
                    )
                    outputs = self.model(front_frames, lat_frames)

                elif self.config.model_type == "multimodal":
                    front_frames, lat_frames, front_pose, lat_pose, labels = (
                        self._extract_multimodal_data(batch)
                    )
                    outputs = self.model(front_frames, lat_frames, front_pose, lat_pose)

                # Compute loss
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                # Store predictions and targets for metrics calculation
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        # Combine batch results
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)

        # Calculate metrics
        metrics = calculate_metrics(all_predictions, all_targets)

        # Calculate average test loss
        metrics["loss"] = test_loss / len(self.test_loader)

        print(f"\nTest Results:")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Hamming Loss: {metrics['hamming_distance']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Jaccard Index: {metrics['jaccard_index']:.4f}")

        # Print per-feature metrics
        print("\nPer-Feature Metrics:")
        for i, feature_metrics in enumerate(metrics["feature_metrics"]):
            print(f"Feature {i}:")
            print(f"  TP: {feature_metrics['TP']}")
            print(f"  TN: {feature_metrics['TN']}")
            print(f"  FP: {feature_metrics['FP']}")
            print(f"  FN: {feature_metrics['FN']}")
            print(f"  Hamming Loss: {feature_metrics['hamming_distance']:.4f}")
            print(f"  F1 Score: {feature_metrics['f1_score']:.4f}")

        return metrics

    def _extract_pose_data(self, batch):
        """Extract pose data from batch for pose model."""
        (
            image_frontal,
            image_lateral,
            pose_frontal,
            pose_lateral,
            label_class,
            ratings,
        ) = batch
        pose_frontal = pose_frontal.to(self.device)
        pose_lateral = pose_lateral.to(self.device)
        ratings = ratings.to(self.device)
        return pose_frontal, pose_lateral, label_class.to(self.device), ratings

    def _extract_vision_data(self, batch):
        """Extract vision data from batch for vision model."""
        (
            image_frontal,
            image_lateral,
            pose_frontal,
            pose_lateral,
            label_class,
            ratings,
        ) = batch
        image_frontal = image_frontal.to(self.device)
        image_lateral = image_lateral.to(self.device)
        ratings = ratings.to(self.device)
        return image_frontal, image_lateral, label_class.to(self.device), ratings

    def _extract_multimodal_data(self, batch):
        """Extract both vision and pose data from batch for multimodal model."""
        (
            image_frontal,
            image_lateral,
            pose_frontal,
            pose_lateral,
            label_class,
            ratings,
        ) = batch
        image_frontal = image_frontal.to(self.device)
        image_lateral = image_lateral.to(self.device)
        pose_frontal = pose_frontal.to(self.device)
        pose_lateral = pose_lateral.to(self.device)
        ratings = ratings.to(self.device)
        return image_frontal, image_lateral, pose_frontal, pose_lateral, ratings

    def _save_model(self, filename):
        """Save the model to disk."""
        save_path = os.path.join(self.config.output_dir, filename)
        torch.save(self.model.state_dict(), save_path)
        return save_path

    def _load_model(self, model_path):
        """Load a model from disk."""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        return self.model

    def _plot_training_history(self):
        """Plot and save training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(self.history["train_loss"], label="Training Loss")
        ax1.plot(self.history["val_loss"], label="Validation Loss")
        ax1.set_title("Loss vs. Epochs")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # Metrics plot
        ax2.plot(self.history["hamming_distance"], label="Hamming Distance")
        ax2.plot(self.history["f1_score"], label="F1 Score")
        ax2.set_title("Metrics vs. Epochs")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Score")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(self.config.output_dir, "training_history.png")
        plt.savefig(save_path)
        plt.close()


def train_model(model, config, train_loader, val_loader, test_loader=None):
    """
    Convenience function to train a model using the Trainer class.

    Args:
        model: PyTorch model to train
        config: TrainingConfig object
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Optional test data loader

    Returns:
        tuple: (trained_model, history_dict, best_model_path)
    """
    trainer = Trainer(model, config, train_loader, val_loader, test_loader)
    history = trainer.train()

    # Find the best model path
    best_model_files = [
        f for f in os.listdir(config.output_dir) if f.startswith("best_model_epoch_")
    ]
    if best_model_files:
        # Get the most recent best model
        best_model_path = os.path.join(config.output_dir, sorted(best_model_files)[-1])
        # Load the best model
        trainer._load_model(best_model_path)
    else:
        best_model_path = os.path.join(config.output_dir, "final_model.pt")

    return model, history, best_model_path


def evaluate_model(model, config, test_loader, model_path=None):
    """
    Evaluate a model on test data.

    Args:
        model: PyTorch model to evaluate
        config: TrainingConfig object
        test_loader: Test data loader
        model_path: Optional path to saved model weights

    Returns:
        dict: Evaluation metrics
    """
    trainer = Trainer(model, config, None, None, test_loader)
    metrics = trainer.test(model_path)
    return metrics

import os
import cv2
import numpy as np

def preprocess_and_save_images(input_dir, output_dir, target_size=(224, 224)):
    """
    Preprocesses images by resizing and saving them as `.npy` files if not already processed.

    This function performs the following steps:
    1. Iterates through the input directory to find `.jpg` and `.png` images.
    2. Checks if an image has already been processed and saved in `.npy` format.
    3. If not processed, it loads, resizes, and saves the image in `.npy` format.
    4. If the image has been processed but has a size mismatch, it is reprocessed.

    Args:
        input_dir (str): Directory containing the raw images.
        output_dir (str): Directory to save the preprocessed images.
        target_size (tuple): Target size for resizing images (default is (224, 224)).

    Raises:
        Exception: If an error occurs while loading a previously processed `.npy` file.

    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                # Construct paths for input image and output preprocessed file
                image_path = os.path.join(root, file)
                file_name = os.path.splitext(file)[0]
                output_file_path = os.path.join(output_dir, f"{file_name}.npy")

                # Check if the file is already processed
                if os.path.exists(output_file_path):
                    try:
                        preprocessed_img = np.load(output_file_path)
                        if preprocessed_img.shape[:2] == target_size:
                            print(f"Skipping {file}, already processed.")
                            continue
                        else:
                            print(f"Reprocessing {file}, size mismatch detected.")
                    except Exception as e:
                        print(f"Error loading {output_file_path}: {e}. Reprocessing.")

                # Load the image
                img = cv2.imread(image_path)
                if img is not None:
                    # Resize the image
                    img_resized = cv2.resize(img, target_size)

                    # Save as .npy file
                    np.save(output_file_path, img_resized)
                    print(f"Processed and saved: {output_file_path}")
                else:
                    print(f"Warning: Could not load image at {image_path}")

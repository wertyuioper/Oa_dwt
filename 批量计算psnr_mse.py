import cv2
import numpy as np
import os


def calculate_mse(image1, image2):
    """
    Calculate the Mean Squared Error (MSE) between two images.

    Parameters:
        image1 (numpy.ndarray): The first image.
        image2 (numpy.ndarray): The second image.

    Returns:
        float: The MSE value.
    """
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions.")

    mse = np.mean((image1 - image2) ** 2)
    return mse


def calculate_psnr(image1, image2):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters:
        image1 (numpy.ndarray): The first image.
        image2 (numpy.ndarray): The second image.

    Returns:
        float: The PSNR value in dB.
    """
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return float('inf')

    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def process_folder(original_folder, processed_folder):
    """
    Calculate MSE and PSNR for all images in the specified folders.

    Parameters:
        original_folder (str): Path to the folder containing original images.
        processed_folder (str): Path to the folder containing processed images.

    Returns:
        None
    """
    original_images = sorted(os.listdir(original_folder))
    processed_images = sorted(os.listdir(processed_folder))

    if len(original_images) != len(processed_images):
        raise ValueError("The number of images in both folders must be the same.")

    for original_image_name, processed_image_name in zip(original_images, processed_images):
        original_image_path = os.path.join(original_folder, original_image_name)
        processed_image_path = os.path.join(processed_folder, processed_image_name)

        original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
        processed_image = cv2.imread(processed_image_path, cv2.IMREAD_COLOR)

        if original_image is None or processed_image is None:
            print(f"Error loading images: {original_image_name} or {processed_image_name}")
            continue

        if original_image.shape != processed_image.shape:
            print(f"Skipping {original_image_name} and {processed_image_name}: Dimension mismatch.")
            continue

        mse = calculate_mse(original_image, processed_image)
        psnr = calculate_psnr(original_image, processed_image)

        print(f"{original_image_name} - {processed_image_name}:")
        print(f"  MSE: {mse}")
        print(f"  PSNR: {psnr} dB\n")


if __name__ == "__main__":
    # Example usage
    original_folder = "original_images_folder"
    processed_folder = "processed_images_folder"

    if not os.path.exists(original_folder):
        raise FileNotFoundError(f"Original folder not found: {original_folder}")

    if not os.path.exists(processed_folder):
        raise FileNotFoundError(f"Processed folder not found: {processed_folder}")

    process_folder(original_folder, processed_folder)

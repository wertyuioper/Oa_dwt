import cv2
import numpy as np


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


if __name__ == "__main__":
    # Example usage
    # Load the images
    original_image_path = "C:/Users\RT\Desktop\DRCNN\dwt\zs\hua_50/noisy_Q.jpg"
    processed_image_path = "C:/Users\RT\Desktop\DRCNN\dwt\zs\zqqq_30zs/traditional_Q.jpg"

    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    processed_image = cv2.imread(processed_image_path, cv2.IMREAD_COLOR)

    if original_image is None or processed_image is None:
        raise FileNotFoundError("One or both images could not be loaded. Check the file paths.")

    # Ensure the images are of the same size
    if original_image.shape != processed_image.shape:
        raise ValueError("Images must have the same dimensions for PSNR and MSE calculation.")

    # Calculate MSE and PSNR
    mse = calculate_mse(original_image, processed_image)
    psnr = calculate_psnr(original_image, processed_image)

    print(f"MSE: {mse}")
    print(f"PSNR: {psnr} dB")

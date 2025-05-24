import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt


def wavelet_transform_and_reconstruction(image_path):
    # Step 1: Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' could not be loaded.")

    # Step 2: Apply Wavelet Transform
    # Perform 3-level decomposition using a discrete wavelet transform (DWT)
    coeffs = pywt.wavedec2(image, wavelet='haar', level=3)

    # coeffs[0] is the approximation (low frequency), coeffs[1:] is the detail (high frequency)
    cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs

    # Step 3: Extract high-frequency components (cH3, cV3, cD3 are the high-frequency details at level 3)
    high_freq_signal = np.abs(cH3) + np.abs(cV3) + np.abs(cD3)

    # Step 4: Flatten the high-frequency coefficients into a 1D signal
    signal = high_freq_signal.flatten()

    # Step 5: Normalize the signal to enhance visibility
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

    # Step 6: Plot the resulting 1D high-frequency signal
    plt.figure(figsize=(10, 4))

    # Main plot
    plt.subplot(2, 1, 1)
    plt.plot(signal, color='blue')
    plt.title("ECG-like 1D Signal from High-Frequency Wavelet Coefficients")
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Magnitude")
    plt.grid(True)

    # Narrower zoomed-in signal plot
    plt.subplot(2, 1, 2)
    start_index = len(signal) // 4
    end_index = start_index + 200  # Zoom in on a portion of the signal
    plt.plot(signal[start_index:end_index], color='red')
    plt.title("Narrower Portion of High-Frequency Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Magnitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Example usage
image_path = "C:/Users\RT\Desktop\DRCNN\dwt\zs\yuzzZ1/noisy_Q.jpg"  # Replace with your image path
wavelet_transform_and_reconstruction(image_path)

# Oa_dwt
To address the issues of high sensitivity to high-frequency noise and the lack of directional adaptability in threshold design associated with traditional wavelet transform-based denoising methods, this paper proposes a novel image denoising approach: Orientation-Aware Discrete Wavelet Transform (Oa-DWT). The proposed method employs discrete wavelet transform (DWT) to perform multi-scale image decomposition, and introduces direction-adaptive thresholding strategies tailored for different high-frequency subbands. Specifically, the method enhances edge preservation in the horizontal subband (LH), optimizes texture recovery in the vertical subband (HL), and adopts a dynamic noise suppression strategy for the diagonal subband (HH). A nonlinear threshold adjustment function is combined with a multi-scale fusion mechanism, and a perceptual loss constraint is introduced during reconstruction to jointly optimize noise suppression and detail preservation. Experimental results demonstrate that using the sym5 wavelet basis with three-level decomposition yields optimal performance, and achieves a PSNR of 29.31 dB and an SSIM of 0.91, which represents a reduction of approximately 12% in MSE compared to conventional methods. Further validation using deep learning models shows that images preprocessed with Oa-DWT improve classification accuracy by 0.8%–1.15% on the Flowers-5 dataset when evaluated using ResNet-50 and ViT architectures. Additionally, attention heat maps indicate a 37% enhancement in the model’s focus on regions of interest. Overall, the proposed method effectively mitigates high-frequency noise residuals and edge artifacts through direction-aware thresholding and multi-scale feature fusion, offering a robust preprocessing solution for complex imaging scenarios.


The perceptual threshold denoising method based on wavelet directional energy is in the dwt folder

oa_dwt下：
a   Calculates the energy in each direction of a single image
b   Sensitivity analysis of various thresholding functions
c   Output the high-pass filter coefficients of each wavelet family

The ViT training noisy dataset (data_set) is available on Baidu Netdisk:
Link: https://pan.baidu.com/s/1SDotGoIjc1uA8TC1lulq2w?pwd=9je2
Access code: 9je2
The original dataset is named data_set, and the denoised version is data_set2.
Please place both datasets in the same folder as the model.

The ResNet-50 training noisy dataset (data_set) is available on Baidu Netdisk:
Link: https://pan.baidu.com/s/1lMwS5HK4BaQE3x9mqa2eBg?pwd=5p6p Access code: 5p6p
The noisy dataset is named flower_data_daisy, and the denoised version is flower_data.
Please place both datasets in the same folder as the model.

The script used for heatmap visualization is 0main_vit.py.

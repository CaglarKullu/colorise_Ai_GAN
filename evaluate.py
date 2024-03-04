from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

def evaluate_performance(original, predicted):
    # Assuming original and predicted images are normalized [0, 1]
    ssim_scores = []
    psnr_scores = []
    for o, p in zip(original, predicted):
        ssim_score = ssim(o, p, data_range=1.0, multichannel=True)
        psnr_score = psnr(o, p, data_range=1.0)
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)
    print(f"Average SSIM: {np.mean(ssim_scores):.4f}")
    print(f"Average PSNR: {np.mean(psnr_scores):.4f}")

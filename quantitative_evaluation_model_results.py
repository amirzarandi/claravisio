import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from ssim import SSIM
from ssim.utils import get_gaussian_kernel
from PIL import Image
import cv2
from utils_stereofog import calculate_model_results
import torch
import torchvision.transforms as transforms
from pathlib import Path
import pytorch_msssim


def calculate_model_results(results_path):

    results_path = os.path.join(results_path, 'test_latest/images')

    # CW-SSIM implementation
    gaussian_kernel_sigma = 1.5
    gaussian_kernel_width = 11
    gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)

    # Indexing the images
    images = [entry for entry in os.listdir(results_path) if 'fake_B' in entry]

    SSIM_scores = []
    CW_SSIM_scores = []
    PSNR_scores = []
    MAE_scores = []
    MSE_scores = []
    Cross_entropy_scores = []
    Pearson_image_correlations = []

    for i, image in enumerate(images):

        clear_image_nonfloat = cv2.imread(os.path.join(results_path, images[i][:-10] + 'real_B' + '.png'))
        fogged_image_nonfloat = cv2.imread(os.path.join(results_path, images[i][:-10] + 'real_A' + '.png'))
        fake_image_nonfloat = cv2.imread(os.path.join(results_path, images[i]))

        # Calculating the Pearson correlation coefficient between the two images
        pearson_corr = np.corrcoef(clear_image_nonfloat.flatten(), fake_image_nonfloat.flatten())[0, 1]
        Pearson_image_correlations.append(pearson_corr)

        # Calculating the SSIM between the fake image and the clear image
        (SSIM_score_reconstruction, SSIM_diff_reconstruction) = structural_similarity(
            clear_image_nonfloat, fake_image_nonfloat, full=True, multichannel=True, channel_axis=2)
        SSIM_scores.append(SSIM_score_reconstruction)

        # Calculating the CW-SSIM between the fake image and the clear image
        CW_SSIM = SSIM(Image.open(os.path.join(results_path, images[i][:-10] + 'real_B' + '.png'))).cw_ssim_value(
            Image.open(os.path.join(results_path, images[i])))
        CW_SSIM_scores.append(CW_SSIM)

        # Calculating PSNR between the fake image and the clear image
        PSNR = peak_signal_noise_ratio(clear_image_nonfloat, fake_image_nonfloat)
        PSNR_scores.append(PSNR)

        # Calculating MAE between the fake image and the clear image
        clear_image_float = clear_image_nonfloat.astype(np.float32) / 255
        fake_image_float = fake_image_nonfloat.astype(np.float32) / 255
        MAE = np.mean(np.abs(clear_image_float - fake_image_float))
        MAE_scores.append(MAE)

        # Calculating MSE between the fake image and the clear image
        MSE = np.mean((clear_image_float - fake_image_float) ** 2)
        MSE_scores.append(MSE)

        # Calculating Cross Entropy between the fake image and the clear image
        Cross_entropy = -np.mean(clear_image_float * np.log(fake_image_float + 1e-12) + 
                                 (1 - clear_image_float) * np.log(1 - fake_image_float + 1e-12))
        Cross_entropy_scores.append(Cross_entropy)

    # Calculate the average values
    mean_SSIM = np.mean(SSIM_scores)
    mean_CW_SSIM = np.mean(CW_SSIM_scores)
    mean_PSNR = np.mean(PSNR_scores)
    mean_MAE = np.mean(MAE_scores)
    mean_MSE = np.mean(MSE_scores)
    mean_Cross_entropy = np.mean(Cross_entropy_scores)
    mean_Pearson = np.mean(Pearson_image_correlations)

    return mean_SSIM, mean_CW_SSIM, mean_PSNR, mean_MAE, mean_MSE, mean_Cross_entropy, mean_Pearson


parser = argparse.ArgumentParser()
parser.add_argument('--results_path', required=True, help='path to the results folder (with subfolder test_latest)')
args = parser.parse_args()
results_path = args.results_path

mean_SSIM, mean_CW_SSIM, mean_PSNR, mean_MAE, mean_MSE, mean_Cross_entropy, mean_Pearson = calculate_model_results(results_path)

print(f"Mean SSIM: {mean_SSIM:.2f}")
print(f"Mean CW-SSIM: {mean_CW_SSIM:.2f}")
print(f"Mean PSNR: {mean_PSNR:.2f}")
print(f"Mean MAE: {mean_MAE:.2f}")
print(f"Mean MSE: {mean_MSE:.2f}")
print(f"Mean Cross Entropy: {mean_Cross_entropy:.2f}")
print(f"Mean Pearson Correlation: {mean_Pearson:.2f}")
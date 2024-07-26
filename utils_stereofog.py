# -*- coding: utf-8 -*-
"""
Created on Tue 15 Aug 2023
Modified on Thurs 25 Jul 2024

Utility functions by Anton Pollak used in the stereofog project
Additional Metrics added by Amir Zarandi

"""
import os
import numpy as np
from skimage.metrics import structural_similarity
from ssim import SSIM
from ssim.utils import get_gaussian_kernel
from pytorch_msssim import ms_ssim
from PIL import Image
import cv2
import re
import torch
import torch.nn as nn # For the custom loss function
from torch import squeeze
from PIL import Image
import torchvision.transforms as T
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.nn import functional as F
import matplotlib.pyplot as plt

# code for detecting the blurriness of an image (https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/)
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_32F).var()

# Code for calculating the MSE between two images (https://www.tutorialspoint.com/how-to-compare-two-images-in-opencv-python)
def image_mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse

# -- Normalized correlation coefficient (NCC) (https://xcdskd.readthedocs.io/en/latest/cross_correlation/cross_correlation_coefficient.html#Application-as-an-Image-Similarity-Measure)
def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data)

def image_ncc(img1, img2):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    img1, img2 :  numpy arrays of same size
    """
    return (1.0/(img1.size-1)) * np.sum(norm_data(img1)*norm_data(img2))

# Code for calculating the MAE between two images
def image_mae(img1, img2):
    return np.mean(np.abs(img1 - img2))

# Function to calculate Cross-Entropy loss
def image_cross_entropy(img1, img2):
    img1_tensor = torch.tensor(img1, dtype=torch.float32)
    img2_tensor = torch.tensor(img2, dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(img1_tensor, img2_tensor).item()

def calculate_model_results(results_path, epoch='latest', epoch_test=False):
    if epoch_test:
        model_name = results_path.split('/')[-2].replace('_epochs', '')
        results_path = os.path.join(results_path, f'{model_name}/test_{epoch}/images')
    else:
        results_path = os.path.join(results_path, f'test_{epoch}/images')

    # CW-SSIM implementation
    gaussian_kernel_sigma = 1.5
    gaussian_kernel_width = 11
    gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)

    # Indexing the images
    images = [entry for entry in os.listdir(results_path) if 'fake_B' in entry]

    Pearson_image_correlations = []
    MSE_scores = []
    NCC_scores = []
    SSIM_scores = []
    CW_SSIM_scores = []
    MS_SSIM_scores = []
    MAE_scores = []
    PSNR_scores = []
    Cross_entropy_scores = []

    print('Calculating scores for model:', results_path.split('/')[-3])

    for i, image in enumerate(images):
        clear_image_nonfloat = cv2.imread(os.path.join(results_path, images[i][:-10] + 'real_B' + '.png'))
        fogged_image_nonfloat = cv2.imread(os.path.join(results_path, images[i][:-10] + 'real_A' + '.png'))
        fake_image_nonfloat = cv2.imread(os.path.join(results_path, images[i]))

        clear_image_gray = cv2.cvtColor(clear_image_nonfloat, cv2.COLOR_BGR2GRAY)
        fake_image_gray = cv2.cvtColor(fake_image_nonfloat, cv2.COLOR_BGR2GRAY)
        Pearson_image_correlation = np.corrcoef(np.asarray(fake_image_gray), np.asarray(clear_image_gray))
        corrImAbs = np.absolute(Pearson_image_correlation)
        Pearson_image_correlations.append(np.mean(corrImAbs))

        MSE_score = image_mse(clear_image_gray, fake_image_gray)
        MSE_scores.append(MSE_score)

        NCC_score = image_ncc(clear_image_gray, fake_image_gray)
        NCC_scores.append(NCC_score)

        (SSIM_score_reconstruction, SSIM_diff_reconstruction) = structural_similarity(clear_image_nonfloat, fogged_image_nonfloat, full=True, multichannel=True, channel_axis=2)
        SSIM_scores.append(SSIM_score_reconstruction)

        CW_SSIM = SSIM(Image.open(os.path.join(results_path, images[i][:-10] + 'real_B' + '.png'))).cw_ssim_value(Image.open(os.path.join(results_path, images[i])))
        CW_SSIM_scores.append(CW_SSIM)

        real_img = np.array(Image.open(os.path.join(results_path, images[i][:-10] + 'real_B' + '.png'))).astype(np.float32)
        real_img = torch.from_numpy(real_img).unsqueeze(0).permute(0, 3, 1, 2)
        fake_img = np.array(Image.open(os.path.join(results_path, images[i]))).astype(np.float32)
        fake_img = torch.from_numpy(fake_img).unsqueeze(0).permute(0, 3, 1, 2)
        MS_SSIM = ms_ssim(real_img, fake_img, data_range=255, size_average=True).item()
        MS_SSIM_scores.append(MS_SSIM)

        # Calculate MAE
        MAE_score = image_mae(clear_image_gray, fake_image_gray)
        MAE_scores.append(MAE_score)

        # Calculate PSNR
        PSNR_score = psnr(clear_image_gray, fake_image_gray, data_range=255)
        PSNR_scores.append(PSNR_score)

        # Calculate Cross-Entropy
        Cross_entropy_score = image_cross_entropy(clear_image_gray, fake_image_gray)
        Cross_entropy_scores.append(Cross_entropy_score)

        if i % 25 == 0:
            print(f'Image {i} of {len(images)}')

    mean_Pearson = np.mean(Pearson_image_correlations)
    mean_MSE = np.mean(MSE_scores)
    mean_NCC = np.mean(NCC_scores)
    mean_SSIM = np.mean(SSIM_scores)
    mean_CW_SSIM = np.mean(CW_SSIM_scores)
    mean_MS_SSIM = np.mean(MS_SSIM_scores)
    mean_MAE = np.mean(MAE_scores)
    mean_PSNR = np.mean(PSNR_scores)
    mean_Cross_entropy = np.mean(Cross_entropy_scores)

    return mean_Pearson, mean_MSE, mean_NCC, mean_SSIM, mean_CW_SSIM, mean_MS_SSIM, mean_MAE, mean_PSNR, mean_Cross_entropy
    
# Code source: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/1161
def generate_stats_from_log(experiment_name, line_interval=10, nb_data=10800, enforce_last_line=True, fig = None, ax = None, highlight_epoch=None):
    """
    Generate chart with all losses from log file generated by CycleGAN/Pix2pix/CUT framework
    """
    #extract every lines
    with open(os.path.join(experiment_name, "loss_log.txt"), 'r') as f:
        lines = f.readlines()
    #choose the lines to use for plotting
    lines_for_plot = []
    for i in range(1,len(lines)):
        if (i-1) % line_interval==0:
            lines_for_plot.append(lines[i])
    if enforce_last_line:
        lines_for_plot.append(lines[-1])
    #initialize dict with loss names
    dicts = dict()
    dicts["epoch"] = []
    parts = (lines_for_plot[0]).split(') ')[1].split(' ')
    for i in range(0, len(parts)//2):
        dicts[parts[2*i][:-1]] = []
    #extract all data
    pattern = "epoch: ([0-9]+), iters: ([0-9]+)"
    for l in lines_for_plot:
        search = re.search(pattern, l)
        epoch = int(search.group(1))
        epoch_floatpart = int(search.group(2))/nb_data
        dicts["epoch"].append(epoch+epoch_floatpart) #to allow several plots for the same epoch
        parts = l.split(') ')[1].split(' ')
        for i in range(0, len(parts)//2):
            dicts[parts[2*i][:-1]].append(float(parts[2*i+1]))
    #plot everything

    if fig is None and ax is None:
        fig, ax = plt.subplots(1,1)
    # plt.figure()
    for key in dicts.keys():
        if key != "epoch":
            ax.plot(dicts["epoch"], dicts[key], label=key)
    ax.legend(loc="best")

    if highlight_epoch is not None:
        ax.scatter(highlight_epoch, dicts['G_GAN'][highlight_epoch])
        ax.scatter(highlight_epoch, dicts['G_L1'][highlight_epoch])
        ax.scatter(highlight_epoch, dicts['D_real'][highlight_epoch])
        ax.scatter(highlight_epoch, dicts['D_fake'][highlight_epoch])

    return fig, ax


image_transform = T.ToPILImage()

class CW_SSIM(nn.Module):
    def __init__(self):
        super(CW_SSIM, self).__init__()



    def forward(self, inputs, targets):
        # print('Shapes:', inputs.min(), inputs.max())
        loss = SSIM(image_transform(squeeze(inputs))).cw_ssim_value(image_transform(squeeze(targets)))

        return loss
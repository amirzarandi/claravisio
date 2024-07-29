from general_imports import *
import argparse
import os
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from utils_stereofog import variance_of_laplacian
from ssim import SSIM
from ssim.utils import get_gaussian_kernel
from pytorch_msssim import ms_ssim
from PIL import Image
import torch

# Set up custom font similar to Calibri
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']  # Arial is similar to Calibri

title_fontsize = 28
label_fontsize = 20

parser = argparse.ArgumentParser()

parser.add_argument('--results_path', required=True, help='path to the results folder (with subfolder test_latest)')
parser.add_argument('--num_images', type=int, default=5, help='number of images to plot')
parser.add_argument('--shuffle', action='store_true', help='if specified, shuffle the images before plotting')
parser.add_argument('--seed', type=int, default=16, help='seed for the random shuffling of the images')
parser.add_argument('--ratio', default=4/3, help='aspect ratio of the images (to undo the transformation from pix2pix model)')
parser.add_argument('--no_laplace', action='store_true', help='if specified, do not plot the Laplacian variance on the fogged images')
parser.add_argument('--no_fog_colorbar', action='store_true', help='if specified, do not plot the fogginess colorbar below')
parser.add_argument('--sort_by_laplacian', action='store_true', help='if specified, sort the images by the Laplacian variance')
parser.add_argument('--specify_image', action='store_true', help='if specified, specify the image to plot')
parser.add_argument('--image_name', default='', help='name of the image to plot')
parser.add_argument('--model_type', type=str, default='pix2pix', help='type of model used (pix2pix or cycleGAN)')
parser.add_argument('--dataset_path', type=str, default='', help='path to the dataset for the cycleGAN evaluation')

args = parser.parse_args()

results_path = args.results_path
num_images = args.num_images
shuffle = args.shuffle
seed = args.seed
ratio = args.ratio
no_laplace = args.no_laplace
no_fog_colorbar = args.no_fog_colorbar
sort_by_laplacian = args.sort_by_laplacian
specify_image = args.specify_image
image_name = args.image_name
model_type = args.model_type.lower()
dataset_path = args.dataset_path

if model_type == 'pix2pix' and dataset_path != '':
    print('The dataset path is not used for the pix2pix model. Continuing...')
elif model_type == 'cyclegan' and dataset_path == '':
    raise ValueError('The dataset path must be specified for the cycleGAN model.')

if no_laplace:
    no_fog_colorbar = True

if type(ratio) != float:
    try:
        ratio = float(ratio.split('/')[0])/float(ratio.split('/')[1])
    except:
        raise ValueError('The ratio must be a float or a string of the form "x/y".')

original_results_path = results_path
results_path = os.path.join(results_path, 'test_latest/images')

# CW-SSIM implementation
gaussian_kernel_sigma = 1.5
gaussian_kernel_width = 11
gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)

# Indexing the images
if specify_image:
    images = [image_name]
    num_images = 1
else:
    if model_type == 'pix2pix':
        images = [entry for entry in os.listdir(results_path) if 'fake_B' in entry]
    elif model_type == 'cyclegan':
        images = [entry for entry in os.listdir(results_path) if 'fake' in entry]
    else:
        raise ValueError('The model type must be either "pix2pix" or "cycleGAN".')

# Defining commonly used variables
if model_type == 'pix2pix':
    real_foggy_image_addition = 'real_A'
    real_clear_image_addition = 'real_B'
    letters_to_remove = 10
elif model_type == 'cyclegan':
    real_foggy_image_addition = 'real'
    dataset_path = os.path.join(dataset_path, 'testA')
    letters_to_remove = 8

# Shuffling the images if specified
if shuffle:
    random.seed(seed)
    random.shuffle(images)

# Setting width and height
width_per_image = 5
height_per_image = width_per_image / ratio

images = images[:num_images]

# Computing the Variance of the Laplacian for each of the images
if not no_laplace:
    laplacian_values = []
    for i in range(num_images):
        fogged_image_gray = cv2.cvtColor(cv2.imread(os.path.join(results_path, images[i][:-letters_to_remove] + real_foggy_image_addition + '.png')), cv2.COLOR_BGR2GRAY)
        laplacian_values += [variance_of_laplacian(fogged_image_gray)]

# Sorting the images by the Laplacian variance if specified
if sort_by_laplacian:
    both = sorted(zip(laplacian_values, images))
    laplacian_values = [laplacian_value for laplacian_value, image in both]
    images = [image for laplacian_value, image in both]

if not no_fog_colorbar or not no_laplace:
    min_fog_value_limit = min(laplacian_values)
    max_fog_value_limit = max(laplacian_values)
    center_fog_value_limit = min_fog_value_limit + (max_fog_value_limit - min_fog_value_limit)*0.4
    norm = plt.Normalize(vmin=min_fog_value_limit, vmax=max_fog_value_limit)

# Create main figure
fig, axes = plt.subplots(num_images, 3, figsize=(3*width_per_image, num_images*height_per_image))
plt.subplots_adjust(hspace=0.3, wspace=0.1)

# Use a more aesthetically pleasing colormap
colormap = cm.viridis

for i in range(num_images):
    # Reading in the images
    img1 = plt.imread(os.path.join(results_path, images[i]))
    img2 = plt.imread(os.path.join(results_path, images[i][:-letters_to_remove] + real_foggy_image_addition + '.png'))
    if model_type == 'pix2pix':
        img3 = plt.imread(os.path.join(results_path, images[i][:-letters_to_remove] + real_clear_image_addition + '.png'))
    elif model_type == 'cyclegan':
        img3 = plt.imread(os.path.join(dataset_path, images[i][:-letters_to_remove-1] + '.png'))

    # Plot images
    axes[i, 0].imshow(img1, aspect='auto')
    axes[i, 1].imshow(img2, aspect='auto')
    axes[i, 2].imshow(img3, aspect='auto')

    # Remove axes
    for ax in axes[i]:
        ax.axis('off')

    # Add titles only to the first row
    if i == 0:
        axes[i, 0].set_title('Reconstructed', fontsize=title_fontsize, pad=20)
        axes[i, 1].set_title('Foggy Real', fontsize=title_fontsize, pad=20)
        axes[i, 2].set_title('Ground Truth', fontsize=title_fontsize, pad=20)

    # Calculate metrics
    fogged_image_gray = cv2.cvtColor(cv2.imread(os.path.join(results_path, images[i][:-letters_to_remove] + real_foggy_image_addition + '.png')), cv2.COLOR_BGR2GRAY)
    fogged_image_nonfloat = cv2.imread(os.path.join(results_path, images[i][:-letters_to_remove] + real_foggy_image_addition + '.png'))
    if model_type == 'pix2pix':
        clear_image_nonfloat = cv2.imread(os.path.join(results_path, images[i][:-letters_to_remove] + real_clear_image_addition + '.png'))
    elif model_type == 'cyclegan':
        h, w, c = fogged_image_nonfloat.shape
        clear_image_nonfloat = cv2.resize(cv2.imread(os.path.join(dataset_path, images[i][:-letters_to_remove-1] + '.png')), (w, h))

    fake_image_nonfloat = cv2.imread(os.path.join(results_path, images[i]))

    # Calculate SSIM
    (SSIM_score, SSIM_diff) = structural_similarity(clear_image_nonfloat, fogged_image_nonfloat, full=True, multichannel=True, channel_axis=2)
    
    # Calculate Pearson correlation
    clear_image_gray = cv2.cvtColor(clear_image_nonfloat, cv2.COLOR_BGR2GRAY)
    Pearson_image_correlation = np.corrcoef(np.asarray(fogged_image_gray), np.asarray(clear_image_gray))
    corrImAbs = np.absolute(Pearson_image_correlation)

    # Calculate SSIM for reconstruction
    (SSIM_score_reconstruction, SSIM_diff_reconstruction) = structural_similarity(clear_image_nonfloat, fake_image_nonfloat, full=True, multichannel=True, channel_axis=2)

    # Calculate CW-SSIM
    if model_type == 'pix2pix':
        CW_SSIM = SSIM(Image.open(os.path.join(results_path, images[i][:-letters_to_remove] + real_clear_image_addition + '.png'))).cw_ssim_value(Image.open(os.path.join(results_path, images[i])))
    elif model_type == 'cyclegan':
        CW_SSIM = 0

    # Calculate MS-SSIM
    if model_type == 'pix2pix':
        real_img = np.array(Image.open(os.path.join(results_path, images[i][:-letters_to_remove] + real_clear_image_addition + '.png'))).astype(np.float32)
    elif model_type == 'cyclegan':
        real_img = np.array(Image.open(os.path.join(dataset_path, images[i][:-letters_to_remove-1] + '.png')).resize((w, h))).astype(np.float32)
    real_img = torch.from_numpy(real_img).unsqueeze(0).permute(0, 3, 1, 2)
    fake_img = np.array(Image.open(os.path.join(results_path, images[i]))).astype(np.float32)
    fake_img = torch.from_numpy(fake_img).unsqueeze(0).permute(0, 3, 1, 2)
    MS_SSIM = ms_ssim(real_img, fake_img, data_range=255, size_average=True).item()

    axes[i, 1].text(0.02, 0.98, f'SSIM: {SSIM_score:.2f}', transform=axes[i, 1].transAxes,
                    color='white', fontsize=label_fontsize, ha='left', va='top', bbox=dict(facecolor='black', alpha=0.7))
    axes[i, 1].text(0.98, 0.98, f'Pearson: {np.mean(corrImAbs):.2f}', transform=axes[i, 1].transAxes,
                    color='white', fontsize=label_fontsize, ha='right', va='top', bbox=dict(facecolor='black', alpha=0.7))

    axes[i, 0].text(0.02, 0.98, f'SSIM (r): {SSIM_score_reconstruction:.2f}', transform=axes[i, 0].transAxes,
                    color='white', fontsize=label_fontsize, ha='left', va='top', bbox=dict(facecolor='black', alpha=0.7))
    axes[i, 0].text(0.98, 0.98, f'CW-SSIM (r): {CW_SSIM:.2f}', transform=axes[i, 0].transAxes,
                    color='white', fontsize=label_fontsize, ha='right', va='top', bbox=dict(facecolor='black', alpha=0.7))
    axes[i, 0].text(0.98, 0.02, f'MS-SSIM (r): {MS_SSIM:.2f}', transform=axes[i, 0].transAxes,
                    color='white', fontsize=label_fontsize, ha='right', va='bottom', bbox=dict(facecolor='black', alpha=0.7))

# Add v_L value inside the foggy image
    v_l_color = colormap(norm(laplacian_values[i]))
    axes[i, 1].text(0.5, 0.02, f'$v_L$: {laplacian_values[i]:.2f}', transform=axes[i, 1].transAxes,
                    color='white', fontsize=label_fontsize, ha='center', va='bottom',
                    bbox=dict(facecolor=v_l_color, edgecolor='white', alpha=0.7, pad=2))

# Adjust the layout to create more space at the bottom
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)  # Adjust this value to increase or decrease bottom margin


# Add colorbar at the bottom
cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), cax=cbar_ax, orientation='horizontal')
cbar.set_label('Fog Intensity ($v_L$)', fontsize=label_fontsize)
cbar.ax.tick_params(labelsize=label_fontsize)



# Save main figure
output_filename = f"{os.path.basename(original_results_path)}_evaluation.png"
plt.savefig(os.path.join(original_results_path, output_filename), dpi=300, bbox_inches='tight')
print(f"Saved the evaluation plot to {os.path.join(original_results_path, output_filename)}.")
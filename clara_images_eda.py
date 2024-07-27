# from general_imports import *
# import matplotlib
# from matplotlib import cm   # colormap for the rating of the fogginess
# from utils_stereofog import variance_of_laplacian
# from PIL import Image
# from ssim import SSIM

# import cv2

# os.chdir("/scratch/general/nfs1/u6059624/ClaraVisio")

# dataset_path = 'claravisio_images'

# variance_values = []
# image_count = 0
# subfolder = 'B'

# for folder in [item for item in os.listdir(dataset_path) if item[0] != '.' and '.xlsx' not in item and '.md' not in item]:
    
#     for image in [item for item in os.listdir(os.path.join(dataset_path, folder, subfolder)) if item[0] != '.']:

#         image_count += 1
#         image_path = os.path.join(dataset_path, folder, subfolder, image)
#         image_gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
#         variance_values.append(variance_of_laplacian(image_gray))

# print('Total images: {}'.format(image_count))
# print('Mean variance: {}'.format(sum(variance_values) / len(variance_values)))

# variance_values_df = pd.Series(sorted(variance_values))

# Q1 = np.percentile(variance_values_df, 25, method='midpoint')
# Q3 = np.percentile(variance_values_df, 75, method='midpoint')
# IQR = Q3 - Q1

# upper=Q3+1.5*IQR
# lower=Q1-1.5*IQR

# print(f'Proportion of datapoints retained: {len(variance_values_df[(variance_values_df < upper) & (variance_values_df > lower)]) / len(variance_values_df)*100:.2f}%')

# fig, ax = plt.subplots(figsize=set_size())

# sns.histplot(variance_values_df[(variance_values_df < upper) & (variance_values_df > lower)], ax = ax, kde=True, fill=True, alpha=0.6)
# ax.set_xlabel('Variance of Laplacian')
# ax.set_ylabel('Count')

# plt.savefig('/uufs/chpc.utah.edu/common/home/u6059624/claravisio/plots/claravisio_dataset_variances.pdf', format='pdf', bbox_inches='tight')

# ax.set_title(f'Variances of Laplacian for Stereofog Dataset ({len(variance_values_df[(variance_values_df < upper) & (variance_values_df > lower)]) / len(variance_values_df)*100:.2f}% of images)')
# plt.show()

# # Select images for low, medium, and high fog
# num_images = len(variance_values)
# low_fog = variance_values[int(num_images * 0.1)]
# medium_fog = variance_values[int(num_images * 0.5)]
# high_fog = variance_values[int(num_images * 0.9)]

# selected_images = [low_fog, medium_fog, high_fog]

# # Create plot
# ratio = 4 / 3
# width_per_image = 4
# height_per_image = width_per_image / ratio
# num_images = 6

# fig = plt.figure(figsize=(num_images*width_per_image, height_per_image))
# ax = [fig.add_subplot(1, num_images, i+1) for i in range(num_images)]

# variances = []
# for index, (variance, img_path) in enumerate(selected_images):
#     img = cv2.imread(img_path)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     ax[index*2].imshow(img_rgb, aspect='auto')
#     ax[index*2].axis('off')
#     ax[index*2+1].imshow(img_rgb, aspect='auto')
#     ax[index*2+1].axis('off')
    
#     variances.append(variance)

# norm = plt.Normalize(vmin=min(variances), vmax=max(variances))

# for index in range(num_images//2):
#     ax[index*2+1].text(0.5, -0.099, f'{variances[index]:.2f}', 
#                        horizontalalignment='center', verticalalignment='top', 
#                        transform=ax[index*2+1].transAxes, fontweight='bold', 
#                        backgroundcolor=cm.jet_r(norm(variances[index])), 
#                        fontsize=40, color='black' if index == 1 else 'white')

# ax[0].text(1., 1.15, 'Low fog', horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes, fontsize=45)
# ax[2].text(1., 1.15, 'Medium fog', horizontalalignment='center', verticalalignment='center', transform=ax[2].transAxes, fontsize=45)
# ax[4].text(1., 1.15, 'High fog', horizontalalignment='center', verticalalignment='center', transform=ax[4].transAxes, fontsize=45)

# plt.subplots_adjust(hspace=0, wspace=0)
# plt.savefig('claravisio_dataset_fog_examples.pdf', format='pdf', bbox_inches='tight')
# plt.show()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from matplotlib import cm

os.chdir("/scratch/general/nfs1/u6059624/ClaraVisio")

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def set_size(width_pt=469.75502, fraction=1, subplots=(1, 1)):
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)

# Set paths
dataset_path = 'claravisio_images'
subfolder = 'B'

# Calculate variance values
variance_values = []
for folder in [item for item in os.listdir(dataset_path) if item[0] != '.' and '.xlsx' not in item and '.md' not in item]:
    for image in [item for item in os.listdir(os.path.join(dataset_path, folder, subfolder)) if item[0] != '.']:
        image_path = os.path.join(dataset_path, folder, subfolder, image)
        image_gray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
        variance_values.append((variance_of_laplacian(image_gray), image_path))

# Sort variance values
variance_values.sort(key=lambda x: x[0])

# Select images for low, medium, and high fog
num_images = len(variance_values)
high_fog = variance_values[int(num_images * 0.1)]
medium_fog = variance_values[int(num_images * 0.5)]
low_fog = variance_values[int(num_images * 0.9)]

selected_images = [low_fog, medium_fog, high_fog]

# Create plot
ratio = 4 / 3
width_per_image = 4
height_per_image = width_per_image / ratio
num_images = 6

fig = plt.figure(figsize=(num_images*width_per_image, height_per_image))
ax = [fig.add_subplot(1, num_images, i+1) for i in range(num_images)]

variances = []
for index, (variance, img_path) in enumerate(selected_images):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    ax[index*2].imshow(img_rgb, aspect='auto')
    ax[index*2].axis('off')
    ax[index*2+1].imshow(img_rgb, aspect='auto')
    ax[index*2+1].axis('off')
    
    variances.append(variance)

norm = plt.Normalize(vmin=min(variances), vmax=max(variances))

for index in range(num_images//2):
    ax[index*2+1].text(0.5, -0.099, f'{variances[index]:.2f}', 
                       horizontalalignment='center', verticalalignment='top', 
                       transform=ax[index*2+1].transAxes, fontweight='bold', 
                       backgroundcolor=cm.jet_r(norm(variances[index])), 
                       fontsize=40, color='black' if index == 1 else 'white')

ax[0].text(1., 1.15, 'Low fog', horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes, fontsize=45)
ax[2].text(1., 1.15, 'Medium fog', horizontalalignment='center', verticalalignment='center', transform=ax[2].transAxes, fontsize=45)
ax[4].text(1., 1.15, 'High fog', horizontalalignment='center', verticalalignment='center', transform=ax[4].transAxes, fontsize=45)

plt.subplots_adjust(hspace=0, wspace=0)
plt.savefig('/uufs/chpc.utah.edu/common/home/u6059624/claravisio/plots/claravisio_dataset_fog_examples.pdf', format='pdf', bbox_inches='tight')
plt.show()
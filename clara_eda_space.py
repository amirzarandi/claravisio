# import os
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from skimage.filters import laplace
# from skimage.metrics import structural_similarity as ssim

# def variance_of_laplacian(image):
#     return cv2.Laplacian(image, cv2.CV_64F).var()

# def spatially_varying_mae(clear_img, foggy_img, window_size=8):
#     h, w = clear_img.shape
#     mae_map = np.zeros((h, w))
    
#     for i in range(0, h, window_size):
#         for j in range(0, w, window_size):
#             clear_window = clear_img[i:i+window_size, j:j+window_size]
#             foggy_window = foggy_img[i:i+window_size, j:j+window_size]
#             mae_map[i:i+window_size, j:j+window_size] = np.mean(np.abs(clear_window - foggy_window))
    
#     return mae_map

# def spatially_varying_laplacian_variance(image, window_size=8):
#     h, w = image.shape
#     var_map = np.zeros((h, w))
    
#     for i in range(0, h, window_size):
#         for j in range(0, w, window_size):
#             window = image[i:i+window_size, j:j+window_size]
#             var_map[i:i+window_size, j:j+window_size] = variance_of_laplacian(window)
    
#     return var_map

# def analyze_dataset(dataset_path, clear_subfolder='A', foggy_subfolder='B'):
#     variance_values = []
#     mae_values = []
#     ssim_values = []
#     image_count = 0

#     for folder in [item for item in os.listdir(dataset_path) if item[0] != '.' and '.xlsx' not in item and '.md' not in item]:
#         clear_folder = os.path.join(dataset_path, folder, clear_subfolder)
#         foggy_folder = os.path.join(dataset_path, folder, foggy_subfolder)
        
#         for image in [item for item in os.listdir(foggy_folder) if item[0] != '.']:
#             image_count += 1
            
#             clear_path = os.path.join(clear_folder, image)
#             foggy_path = os.path.join(foggy_folder, image)
            
#             clear_img = cv2.imread(clear_path, cv2.IMREAD_GRAYSCALE)
#             foggy_img = cv2.imread(foggy_path, cv2.IMREAD_GRAYSCALE)
            
#             # Variance of Laplacian
#             variance_values.append(variance_of_laplacian(foggy_img))
            
#             # Spatially varying MAE
#             mae_map = spatially_varying_mae(clear_img, foggy_img)
#             mae_values.append(np.mean(mae_map))
            
#             # SSIM
#             ssim_value, _ = ssim(clear_img, foggy_img, full=True)
#             ssim_values.append(ssim_value)
            
#             # Spatially varying Laplacian variance
#             var_map = spatially_varying_laplacian_variance(foggy_img)
            
#             # You can save or further analyze mae_map and var_map here
            
#             if image_count % 100 == 0:
#                 print(f"Processed {image_count} images")

#     results = {
#         'total_images': image_count,
#         'mean_variance': np.mean(variance_values),
#         'std_variance': np.std(variance_values),
#         'mean_mae': np.mean(mae_values),
#         'std_mae': np.std(mae_values),
#         'mean_ssim': np.mean(ssim_values),
#         'std_ssim': np.std(ssim_values),
#         'variance_values': variance_values,
#         'mae_values': mae_values,
#         'ssim_values': ssim_values
#     }

#     return results

# def plot_histogram(data, title, xlabel, ylabel, filename, save_dir):
#     plt.figure(figsize=(10, 6))
#     plt.hist(data, bins=50, edgecolor='black')
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
    
#     # Ensure the save directory exists
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Save the plot to the specified directory
#     full_path = os.path.join(save_dir, filename)
#     plt.savefig(full_path)
#     plt.close()
    
#     print(f"Saved histogram: {full_path}")

# # Main execution
# if __name__ == "__main__":
#     os.chdir("/scratch/general/nfs1/u6059624/StereoFog")
#     dataset_path = 'stereofog_images'
#     plots_dir = "/uufs/chpc.utah.edu/common/home/u6059624/claravisio/plots"
    
#     results = analyze_dataset(dataset_path)
    
#     print("Analysis Results:")
#     for key, value in results.items():
#         if not isinstance(value, list):  # Only print scalar values
#             print(f"{key}: {value}")
    
#     # Now we can create the histograms
#     plot_histogram(results['variance_values'], "Distribution of Laplacian Variance", "Variance", "Frequency", "laplacian_variance_hist_stereo.png", plots_dir)
#     plot_histogram(results['mae_values'], "Distribution of MAE", "MAE", "Frequency", "mae_hist_stereo.png", plots_dir)
#     plot_histogram(results['ssim_values'], "Distribution of SSIM", "SSIM", "Frequency", "ssim_hist_stereo.png", plots_dir)

#     print(f"Histograms have been saved in the directory: {plots_dir}")

import numpy as np
import cv2
from scipy.stats import kurtosis
from skimage.feature import canny
import os
import matplotlib.pyplot as plt

def normalize_image_size(image, target_size=(256, 256)):
    """
    Resize the image to a standard size to ensure resolution independence.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def local_blur_estimation(image, kernel_size=9):
    """
    Estimate local blur using variance of Laplacian in small patches.
    Kernel size is relative to image size.
    """
    h, w = image.shape
    relative_kernel_size = max(3, int(min(h, w) * kernel_size / 256))
    relative_kernel_size = relative_kernel_size if relative_kernel_size % 2 == 1 else relative_kernel_size + 1
    
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    blur_map = cv2.blur(np.abs(laplacian), (relative_kernel_size, relative_kernel_size))
    return blur_map

def edge_density(image, sigma=2.0):
    """
    Compute edge density using Canny edge detection.
    Sigma is relative to image size.
    """
    h, w = image.shape
    relative_sigma = sigma * min(h, w) / 256
    edges = canny(image, sigma=relative_sigma)
    return np.mean(edges)

def non_homogeneity_measure(image):
    """
    Compute a measure of non-homogeneity in blur.
    """
    # Normalize image size
    norm_image = normalize_image_size(image)
    
    # Step 1: Local blur estimation
    blur_map = local_blur_estimation(norm_image)
    
    # Step 2: Global statistics
    blur_mean = np.mean(blur_map)
    blur_std = np.std(blur_map)
    blur_kurtosis = kurtosis(blur_map.flatten())
    
    # Step 3: Edge analysis
    edge_dens = edge_density(norm_image)
    
    # Combine measures
    non_homogeneity = blur_std / blur_mean  # Coefficient of variation
    sharpness = edge_dens * blur_kurtosis
    
    return {
        'non_homogeneity': non_homogeneity,
        'sharpness': sharpness,
        'blur_mean': blur_mean,
        'blur_std': blur_std,
        'blur_kurtosis': blur_kurtosis,
        'edge_density': edge_dens
    }

def analyze_dataset(dataset_path, foggy_subfolder='B'):
    results = []
    image_count = 0

    for folder in [item for item in os.listdir(dataset_path) if item[0] != '.' and '.xlsx' not in item and '.md' not in item]:
        foggy_folder = os.path.join(dataset_path, folder, foggy_subfolder)
        
        for image in [item for item in os.listdir(foggy_folder) if item[0] != '.']:
            image_count += 1
            
            foggy_path = os.path.join(foggy_folder, image)
            foggy_img = cv2.imread(foggy_path, cv2.IMREAD_GRAYSCALE)
            
            # Analyze non-homogeneous blur
            result = non_homogeneity_measure(foggy_img)
            results.append(result)
            
            if image_count % 100 == 0:
                print(f"Processed {image_count} images")

    # Aggregate results
    aggregate = {key: np.mean([r[key] for r in results]) for key in results[0].keys()}
    aggregate['total_images'] = image_count

    return aggregate

def plot_results(results, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Remove 'total_images' from the plot
    plot_results = {k: v for k, v in results.items() if k != 'total_images'}
    
    bars = ax.bar(plot_results.keys(), plot_results.values())
    
    ax.set_ylabel('Value')
    ax.set_title('Non-Homogeneous Blur Analysis Results')
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved analysis results plot: {save_path}")

# Main execution
if __name__ == "__main__":
    os.chdir("/scratch/general/nfs1/u6059624/StereoFog")
    dataset_path = 'stereofog_images'
    plots_dir = "/uufs/chpc.utah.edu/common/home/u6059624/claravisio/plots"
    
    results = analyze_dataset(dataset_path)
    
    print("Analysis Results:")
    for key, value in results.items():
        print(f"{key}: {value}")
    
    # Ensure the plots directory exists
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot and save the results
    plot_results(results, os.path.join(plots_dir, 'non_homogeneous_blur_analysis_stereo.png'))

    print(f"Non-homogeneous blur analysis plot has been saved in the directory: {plots_dir}")
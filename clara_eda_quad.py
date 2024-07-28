import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def analyze_quadrants(image, num_quadrants=(3, 4)):
    h, w = image.shape
    qh, qw = h // num_quadrants[0], w // num_quadrants[1]
    quadrant_variances = np.zeros(num_quadrants)
    
    for i in range(num_quadrants[0]):
        for j in range(num_quadrants[1]):
            quadrant = image[i*qh:(i+1)*qh, j*qw:(j+1)*qw]
            quadrant_variances[i, j] = variance_of_laplacian(quadrant)
    
    return quadrant_variances

def analyze_dataset(dataset_path, foggy_subfolder='B'):
    quadrant_variances = []
    image_count = 0

    for folder in [item for item in os.listdir(dataset_path) if item[0] != '.' and '.xlsx' not in item and '.md' not in item]:
        foggy_folder = os.path.join(dataset_path, folder, foggy_subfolder)
        
        for image in [item for item in os.listdir(foggy_folder) if item[0] != '.']:
            image_count += 1
            
            foggy_path = os.path.join(foggy_folder, image)
            foggy_img = cv2.imread(foggy_path, cv2.IMREAD_GRAYSCALE)
            
            # Analyze quadrants
            quadrant_vars = analyze_quadrants(foggy_img)
            quadrant_variances.append(quadrant_vars)
            
            if image_count % 100 == 0:
                print(f"Processed {image_count} images")

    quadrant_variances = np.array(quadrant_variances)
    quadrant_means = np.mean(quadrant_variances, axis=0)
    quadrant_stds = np.std(quadrant_variances, axis=0)

    return {
        'total_images': image_count,
        'quadrant_means': quadrant_means,
        'quadrant_stds': quadrant_stds
    }

def plot_quadrant_analysis(means, stds, save_path):
    fig, ax = plt.subplots(figsize=(12, 9))
    
    num_rows, num_cols = means.shape
    for i in range(num_rows):
        for j in range(num_cols):
            rect = plt.Rectangle((j, num_rows-1-i), 1, 1, fill=False)
            ax.add_patch(rect)
            ax.text(j+0.5, num_rows-1-i+0.5, f'Mean: {means[i,j]:.2f}\nStd: {stds[i,j]:.2f}', 
                    ha='center', va='center', wrap=True)

    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Quadrant Analysis: Mean and Std of Laplacian Variance')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Saved quadrant analysis plot: {save_path}")

# Main execution
if __name__ == "__main__":
    os.chdir("/scratch/general/nfs1/u6059624/StereoFog")
    dataset_path = 'stereofog_images'
    plots_dir = "/uufs/chpc.utah.edu/common/home/u6059624/claravisio/plots"
    
    results = analyze_dataset(dataset_path)
    
    print("Analysis Results:")
    print(f"Total images processed: {results['total_images']}")
    
    # Ensure the plots directory exists
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot and save the quadrant analysis
    plot_quadrant_analysis(results['quadrant_means'], results['quadrant_stds'], 
                           os.path.join(plots_dir, 'quadrant_analysis_stereo.png'))

    print(f"Quadrant analysis plot has been saved in the directory: {plots_dir}")
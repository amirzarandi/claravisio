import argparse
from utils_stereofog import calculate_model_results

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', required=True, help='path to the results folder (with subfolder test_latest)')
args = parser.parse_args()
results_path = args.results_path

mean_Pearson, mean_MSE, mean_NCC, mean_SSIM, mean_CW_SSIM, mean_MS_SSIM, mean_MAE, mean_PSNR, mean_Cross_entropy = calculate_model_results(results_path)

print(f"Mean Pearson Correlation: {mean_Pearson:.4f}")
print(f"Mean MSE: {mean_MSE:.4f}")
print(f"Mean NCC: {mean_NCC:.4f}")
print(f"Mean SSIM: {mean_SSIM:.4f}")
print(f"Mean CW-SSIM: {mean_CW_SSIM:.4f}")
print(f"Mean MS-SSIM: {mean_MS_SSIM:.4f}")
print(f"Mean MAE: {mean_MAE:.4f}")
print(f"Mean PSNR: {mean_PSNR:.4f}")
print(f"Mean Cross Entropy: {mean_Cross_entropy:.4f}")
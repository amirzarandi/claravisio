import shutil
import os
import random
import argparse
import subprocess
import Augmentor
import warnings
import sys

parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders for each of the recording runs with A and B subfolders)')
parser.add_argument('--augment', action='store_true', help='whether to augment the dataset or not')

args = parser.parse_args()
path = args.dataroot
augment = args.augment

if 'augmented' not in path and augment:
    warnings.warn("The dataset path does not contain the word 'augmented'. If you are trying to augment the dataset, consider adding 'augment' or some other flag to the name to be able to distinguish it.")
    user_input = input("Are you sure you want to proceed? (y/n): ")
    if user_input == 'n':
        sys.exit()

# Create new processed directory
processed_path = os.path.join(os.path.dirname(path), 'stereofog_images_processed')
os.makedirs(processed_path, exist_ok=True)

# Move all subsets of the dataset into one folder in the processed directory
print('Consolidating dataset into one folder...')
folders = [item for item in os.listdir(path) if item[0] != '.' and item != 'A' and item != 'B' and os.path.isdir(os.path.join(path, item))]

for folder in folders:
    for subfolder in ['A', 'B']:
        new_subfolder_path = os.path.join(processed_path, subfolder)
        os.makedirs(new_subfolder_path, exist_ok=True)

        files = [item for item in os.listdir(os.path.join(path, folder, subfolder)) if item[0] != '.']
        for file in files:
            shutil.copy(os.path.join(path, folder, subfolder, file), os.path.join(new_subfolder_path, file))

print('Creating test split...')
all_files = os.listdir(os.path.join(processed_path, 'A'))

random.seed(0)

subset_test = all_files

for folder in ['/A', '/B']:
    test_folder_path = os.path.join(processed_path, folder[1:], 'test')
    os.makedirs(test_folder_path, exist_ok=True)

    for file in subset_test:
        try:
            shutil.move(os.path.join(processed_path, folder[1:], file), os.path.join(test_folder_path, file))
        except FileNotFoundError:
            pass

if augment:
    print('Augmenting the dataset...')
    p = Augmentor.Pipeline(os.path.join(processed_path, 'B/test'))
    p.ground_truth(os.path.join(processed_path, 'A/test'))
    p.flip_left_right(probability=1)
    p.zoom_random(probability=0.3, percentage_area=0.8)
    p.process()

    augmentor_subfolder = 'B/test/output'
    for image in [item for item in os.listdir(os.path.join(processed_path, augmentor_subfolder)) if not os.path.isdir(os.path.join(processed_path, augmentor_subfolder, item))]:
        image_new = image.rsplit('_', 1)[0]

        if 'groundtruth' in image_new:
            image_new = image_new.replace('_groundtruth_(1)_test_', '').replace('.bmp', '_augmented.bmp')
            try:
                shutil.move(os.path.join(processed_path, augmentor_subfolder, image), os.path.join(processed_path, 'A/test', image_new))
            except FileNotFoundError:
                pass
        else:
            image_new = image_new.replace('test_original_', '').replace('.bmp', '_augmented.bmp')
            try:
                shutil.move(os.path.join(processed_path, augmentor_subfolder, image), os.path.join(processed_path, 'B/test', image_new))
            except FileNotFoundError:
                pass

combination_command = f"python datasets/combine_A_and_B.py --fold_A {os.path.join(processed_path, 'A')} --fold_B {os.path.join(processed_path, 'B')} --fold_AB {processed_path} --no_multiprocessing"
print('Combining the A and B folders...')
subprocess.call(combination_command, shell=True)

print("Dataset preprocessing complete.")
if augment:
    print(f"The dataset contains {len(subset_test) * 2} test images after augmentation.")
else:
    print(f"The dataset contains {len(subset_test)} test images.")

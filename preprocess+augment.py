import shutil
import os
import random
import numpy as np
import argparse
import subprocess
import Augmentor
import warnings
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders for each of the recording runs with A and B subfolders)')
path = parser.parse_args().dataroot

# Create clara_processed and clara_augmented folders
parent_dir = os.path.dirname(path)
clara_processed = os.path.join(parent_dir, 'clara_processed')
clara_augmented = os.path.join(parent_dir, 'clara_augmented')

for folder in [clara_processed, clara_augmented]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def process_data(input_path, output_path):
    # Copy data from input_path to output_path
    shutil.copytree(input_path, output_path, dirs_exist_ok=True)
    
    ## Moving all subsets of dataset into one folder
    print(f'Consolidating dataset into one folder for {output_path}...')
    folders = [item for item in os.listdir(output_path) if item[0] != '.' and item != 'A' and item != 'B' and os.path.isdir(os.path.join(output_path, item))]

    for folder in folders:
        for subfolder in ['A', 'B']:
            if subfolder not in os.listdir(output_path):
                os.mkdir(os.path.join(output_path, subfolder))

            files = [item for item in os.listdir(os.path.join(output_path, folder, subfolder)) if item[0] != '.']

            for file in files:
                os.rename(os.path.join(output_path, folder, subfolder, file), os.path.join(output_path, subfolder, file))

        shutil.rmtree(os.path.join(output_path, folder))

    ## Creating train/test/val splits
    print(f'Creating train/test/val splits for {output_path}...')
    all_files = os.listdir(os.path.join(output_path, 'A'))

    random.seed(0)

    subset_train = random.sample(all_files, round(len(all_files)*0.8))
    remaining = list(set(all_files) - set(subset_train))
    subset_val = random.sample(remaining, round(len(remaining)*0.5))
    subset_test = list(set(remaining) - set(subset_val))

    subfolders = ['train', 'val', 'test']

    for folder in ['A', 'B']:
        for subfolder, subset in zip(subfolders, [subset_train, subset_val, subset_test]):
            os.makedirs(os.path.join(output_path, folder, subfolder), exist_ok=True)

            for file in subset:
                file_name = os.path.join(output_path, folder, file)
                new_file_name = os.path.join(output_path, folder, subfolder, file)
                os.rename(file_name, new_file_name)

    return subset_train, subset_val, subset_test

def augment_data(path):
    print('Augmenting the dataset...')
    p = Augmentor.Pipeline(os.path.join(path, 'B', 'train'))

    # Adding the ground truth data to be transformed
    p.ground_truth(os.path.join(path, 'A', 'train'))
    # Adding operations to the pipeline
    p.flip_left_right(probability=1)
    p.zoom_random(probability=0.3, percentage_area=0.8)
    p.process()  # Transforming each image exactly once

    # Moving the generated images out of the output folder to rejoin them with the original images
    augmentor_subfolder = os.path.join('B', 'train', 'output')

    for image in [item for item in os.listdir(os.path.join(path, augmentor_subfolder)) if not os.path.isdir(os.path.join(path, augmentor_subfolder, item))]:
        image_new = image.rsplit('_', 1)[0]

        if 'groundtruth' in image_new:
            image_new = image_new.replace('_groundtruth_(1)_train_', '').replace('.bmp', '_augmented.bmp')
            os.rename(os.path.join(path, augmentor_subfolder, image), os.path.join(path, 'A', 'train', image_new))
        else:
            image_new = image_new.replace('train_original_', '').replace('.bmp', '_augmented.bmp')
            os.rename(os.path.join(path, augmentor_subfolder, image), os.path.join(path, 'B', 'train', image_new))

def combine_AB(path):
    combination_command = f"python combine_A_and_B.py --fold_A {path}/A --fold_B {path}/B --fold_AB {path} --no_multiprocessing"
    print(f'Combining the A and B folders for {path}...')
    subprocess.call(combination_command, shell=True)

# Process data for clara_processed
subset_train, subset_val, subset_test = process_data(path, clara_processed)
combine_AB(clara_processed)

# Process and augment data for clara_augmented
process_data(path, clara_augmented)
augment_data(clara_augmented)
combine_AB(clara_augmented)

print("Dataset preprocessing complete.")
print(f"The clara_processed dataset contains {len(subset_train)} training images, {len(subset_val)} validation images, and {len(subset_test)} test images.")
print(f"The clara_augmented dataset contains {len(subset_train)*2} training images, {len(subset_val)} validation images, and {len(subset_test)} test images after augmentation.")
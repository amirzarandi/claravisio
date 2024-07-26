import os
import argparse
from PIL import Image

def convert_and_resize_images(input_folder, output_folder, new_size=(320, 240)):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Walk through all subdirectories in the input folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".png"):
                # Full path to the input file
                input_file_path = os.path.join(root, file)

                # Create the corresponding subfolder in the output directory
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)

                # Full path to the output file
                output_file_path = os.path.join(output_subfolder, os.path.splitext(file)[0] + '.bmp')

                # Open, resize, and save the image
                with Image.open(input_file_path) as img:
                    img_resized = img.resize(new_size)
                    img_resized.save(output_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert and resize images.')
    parser.add_argument('--dataroot', type=str, required=True, help='The root directory containing the input images.')
    parser.add_argument('--output_name', type=str, default='bmp_images', help='Name of the output folder (default: bmp_images)')
    args = parser.parse_args()

    input_folder = args.dataroot
    parent_dir = os.path.dirname(input_folder)
    output_folder = os.path.join(parent_dir, args.output_name)

    convert_and_resize_images(input_folder, output_folder)
    print(f"Conversion complete. Output saved to: {output_folder}")
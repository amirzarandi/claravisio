import os
from PIL import Image

def convert_and_resize_images(input_folder, output_folder, new_size=(240, 320)):
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


input_folder = 'datasets/clara_images_processed'
output_folder = 'datasets/clara_images_processed_bmp'

convert_and_resize_images(input_folder, output_folder)

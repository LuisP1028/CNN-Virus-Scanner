import os
import numpy as np
import imageio
from tqdm import tqdm
##########################################################
#This script converts files to grayscale images. These images are 
#used to train/test CNN models.
##########################################################
def file_to_square_image(input_path, output_dir, img_size=256):
    try:
        with open(input_path, 'rb') as f:
            file_bytes = f.read()
        
        target_bytes = img_size * img_size
        bytes_len = len(file_bytes)
        
        if bytes_len < target_bytes:
            padded = file_bytes + bytes([0] * (target_bytes - bytes_len))
            img_data = np.frombuffer(padded, dtype=np.uint8)
        else:
            img_data = np.frombuffer(file_bytes[:target_bytes], dtype=np.uint8)
        
        img_data = img_data.reshape((img_size, img_size))
        output_path = os.path.join(output_dir, f"{os.path.basename(input_path)}.png")
        imageio.imwrite(output_path, img_data)
        return True
    
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def batch_convert_to_images(input_dir, output_dir, img_size=256):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    processed = 0
    file_list = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    with tqdm(total=len(file_list), desc="Converting files") as pbar:
        for filename in file_list:
            input_path = os.path.join(input_dir, filename)
            if file_to_square_image(input_path, output_dir, img_size):
                processed += 1
            pbar.update(1)
    
    print(f"Successfully converted {processed}/{len(file_list)} files")

if __name__ == "__main__":
    INPUT_DIRECTORY = r""
    OUTPUT_DIRECTORY = r""
    IMAGE_SIZE = 256
    
    batch_convert_to_images(INPUT_DIRECTORY, OUTPUT_DIRECTORY, IMAGE_SIZE)
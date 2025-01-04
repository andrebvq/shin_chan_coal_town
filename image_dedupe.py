import os
from PIL import Image
import imagehash
from pathlib import Path
from collections import defaultdict

def deduplicate_images(input_folder):
    # Create output folder name by adding "_deduped"
    output_folder = str(Path(input_folder).absolute()) + "_deduped"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Dictionary to store image hashes
    hash_dict = defaultdict(list)
    
    # Process all PNG files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            filepath = os.path.join(input_folder, filename)
            try:
                # Open image and compute its hash
                with Image.open(filepath) as img:
                    # Using average hash with increased size for better accuracy
                    hash_value = str(imagehash.average_hash(img, hash_size=8))
                    hash_dict[hash_value].append(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Copy unique images to output folder
    unique_count = 0
    duplicate_count = 0
    
    for hash_value, filenames in hash_dict.items():
        # Keep only the first image from each group of duplicates
        original = filenames[0]
        duplicates = filenames[1:]
        
        # Copy the unique image
        src_path = os.path.join(input_folder, original)
        dst_path = os.path.join(output_folder, original)
        try:
            with Image.open(src_path) as img:
                img.save(dst_path)
            unique_count += 1
            if duplicates:
                duplicate_count += len(duplicates)
                print(f"Found duplicates of {original}: {', '.join(duplicates)}")
        except Exception as e:
            print(f"Error copying {original}: {e}")
    
    print(f"\nProcess complete!")
    print(f"Found {unique_count} unique images")
    print(f"Found {duplicate_count} duplicate images")
    print(f"Unique images saved to: {output_folder}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get input folder from user
    folder_path = os.path.join(current_dir, "screenshots")
    deduplicate_images(folder_path)
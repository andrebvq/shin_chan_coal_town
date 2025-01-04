import os
from PIL import Image
import imagehash
from pathlib import Path
from collections import defaultdict
import shutil

def create_directory_structure():
    dirs = [
        "screenshots_processed",
        "screenshots_processed/chara_dialogues",
        "screenshots_processed/box_dialogues",
        "screenshots_processed/chara_dialogues/discarded",
        "screenshots_processed/box_dialogues/discarded"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def organize_screenshots():
    stats = defaultdict(int)
    
    for filename in os.listdir("screenshots"):
        if filename.lower().endswith('.png'):
            source = os.path.join("screenshots", filename)
            if filename.startswith("dialogue_box"):
                destination = os.path.join("screenshots_processed/box_dialogues", filename)
                stats["box_dialogues"] += 1
            else:
                destination = os.path.join("screenshots_processed/chara_dialogues", filename)
                stats["chara_dialogues"] += 1
            shutil.copy2(source, destination)
    
    return stats

def deduplicate_images(input_folder):
    hash_dict = defaultdict(list)
    stats = {
        "unique": 0,
        "duplicates": 0
    }
    
    # Get all PNG files and their hashes
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            filepath = os.path.join(input_folder, filename)
            try:
                with Image.open(filepath) as img:
                    hash_value = str(imagehash.average_hash(img, hash_size=8))
                    hash_dict[hash_value].append(filename)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Move duplicates to discarded folder
    discarded_folder = os.path.join(input_folder, "discarded")
    
    for hash_value, filenames in hash_dict.items():
        if len(filenames) > 1:
            # Keep the first file, move others to discarded
            original = filenames[0]
            duplicates = filenames[1:]
            stats["duplicates"] += len(duplicates)
            
            for duplicate in duplicates:
                src_path = os.path.join(input_folder, duplicate)
                dst_path = os.path.join(discarded_folder, duplicate)
                try:
                    shutil.move(src_path, dst_path)
                except Exception as e:
                    print(f"Error moving {duplicate}: {e}")
        
        stats["unique"] += 1
    
    return stats

def main():
    # Create directory structure
    create_directory_structure()
    
    # Organize screenshots
    org_stats = organize_screenshots()
    print("\nOrganization Statistics:")
    print(f"Character dialogues: {org_stats['chara_dialogues']}")
    print(f"Box dialogues: {org_stats['box_dialogues']}")
    
    # Deduplicate both folders
    folders = [
        "screenshots_processed/chara_dialogues",
        "screenshots_processed/box_dialogues"
    ]
    
    for folder in folders:
        print(f"\nProcessing {os.path.basename(folder)}:")
        stats = deduplicate_images(folder)
        print(f"Unique images: {stats['unique']}")
        print(f"Duplicates moved: {stats['duplicates']}")

if __name__ == "__main__":
    main()
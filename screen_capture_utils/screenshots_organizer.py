import os
from PIL import Image
import imagehash
from pathlib import Path
from collections import defaultdict
import shutil
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated images

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for image analysis
DIALOGUE_BOX_PARAMS = {
    'brightness_range': (180, 240),
    'std_dev_range': (15, 60),
    'brown_text_rgb': ((60, 100), (30, 70), (0, 40)),
    'min_text_ratio': 0.01
}

def validate_image(image_path: str) -> bool:
    """
    Validate if the image is properly formatted and not corrupted.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        bool: True if image is valid, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        logger.error(f"Invalid image {image_path}: {str(e)}")
        return False

def get_processed_dir_name() -> str:
    """Generate the processed directory name with today's date"""
    today = datetime.now().strftime('%Y%m%d')
    return f"screenshots_processed_{today}"

def create_directory_structure() -> str:
    """
    Create the directory structure for organizing screenshots.
    
    Returns:
        str: Path to the processed directory
    """
    try:
        base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        processed_dir = base_dir / get_processed_dir_name()
        
        # Define required subdirectories
        subdirs = [
            "chara_dialogues/discarded",
            "box_dialogues/discarded",
            "no_dialogue"
        ]
        
        # Create all directories
        for subdir in subdirs:
            (processed_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        return str(processed_dir)
    except OSError as e:
        logger.error(f"Failed to create directory structure: {str(e)}")
        raise

def analyze_image_region(region: Image.Image) -> Tuple[float, float, int]:
    """
    Analyze an image region for dialogue box characteristics.
    
    Args:
        region: PIL Image region to analyze
        
    Returns:
        Tuple[float, float, int]: (avg_brightness, std_dev, brown_text_pixels)
    """
    # Convert to grayscale for brightness analysis
    gray_region = region.convert('L')
    pixels = np.array(gray_region)
    
    avg_brightness = np.mean(pixels)
    std_dev = np.std(pixels)
    
    # Check for brown text pixels
    rgb_pixels = np.array(region)
    brown_mask = (
        (rgb_pixels[:,:,0] > DIALOGUE_BOX_PARAMS['brown_text_rgb'][0][0]) &
        (rgb_pixels[:,:,0] < DIALOGUE_BOX_PARAMS['brown_text_rgb'][0][1]) &
        (rgb_pixels[:,:,1] > DIALOGUE_BOX_PARAMS['brown_text_rgb'][1][0]) &
        (rgb_pixels[:,:,1] < DIALOGUE_BOX_PARAMS['brown_text_rgb'][1][1]) &
        (rgb_pixels[:,:,2] > DIALOGUE_BOX_PARAMS['brown_text_rgb'][2][0]) &
        (rgb_pixels[:,:,2] < DIALOGUE_BOX_PARAMS['brown_text_rgb'][2][1])
    )
    brown_text_pixels = np.sum(brown_mask)
    
    return avg_brightness, std_dev, brown_text_pixels

def has_dialogue_box(image_path: str) -> bool:
    """
    Check if the image contains a dialogue box using multiple criteria.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        bool: True if a dialogue box is detected, False otherwise
    """
    try:
        if not validate_image(image_path):
            return False
            
        with Image.open(image_path) as img:
            if img.mode == 'RGBA':
                img = img.convert('RGB')
                
            width, height = img.size
            
            # Define regions to check
            regions = [
                (0.2, 0.3, 0.8, 0.7),  # Center
                (0.1, 0.1, 0.9, 0.3),  # Top
                (0.2, 0.6, 0.8, 0.9)   # Bottom
            ]
            
            for region_bounds in regions:
                left, top, right, bottom = [int(x * (width if i % 2 == 0 else height))
                                          for i, x in enumerate(region_bounds)]
                region = img.crop((left, top, right, bottom))
                
                avg_brightness, std_dev, brown_text_pixels = analyze_image_region(region)
                total_pixels = region.size[0] * region.size[1]
                
                if (DIALOGUE_BOX_PARAMS['brightness_range'][0] < avg_brightness < 
                    DIALOGUE_BOX_PARAMS['brightness_range'][1] and
                    DIALOGUE_BOX_PARAMS['std_dev_range'][0] < std_dev < 
                    DIALOGUE_BOX_PARAMS['std_dev_range'][1] and
                    brown_text_pixels > total_pixels * DIALOGUE_BOX_PARAMS['min_text_ratio']):
                    return True
                    
            return False
                
    except Exception as e:
        logger.error(f"Error analyzing {image_path}: {str(e)}")
        return False

def safe_file_move(src: str, dst: str) -> bool:
    """
    Safely move a file with proper error handling.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        
        # Use copy2 to preserve metadata, then remove source
        shutil.copy2(src, dst)
        os.remove(src)
        return True
    except OSError as e:
        logger.error(f"Failed to move file {src} to {dst}: {str(e)}")
        return False

def calculate_image_hashes(image_path: str) -> Optional[Tuple[imagehash.ImageHash, ...]]:
    """
    Calculate multiple types of perceptual hashes for more accurate duplicate detection.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Optional[Tuple[ImageHash, ...]]: Tuple of image hashes or None if processing fails
    """
    try:
        if not validate_image(image_path):
            return None
            
        with Image.open(image_path) as img:
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # Normalize image size for consistent hashing
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
            
            return (
                imagehash.average_hash(img, hash_size=16),  # Increased hash size
                imagehash.dhash(img, hash_size=16),
                imagehash.phash(img, hash_size=16),
                imagehash.whash(img)  # Added Wavelet hashing
            )
    except Exception as e:
        logger.error(f"Error calculating hash for {image_path}: {str(e)}")
        return None

def are_images_similar(hash1_tuple: Tuple[imagehash.ImageHash, ...],
                      hash2_tuple: Tuple[imagehash.ImageHash, ...],
                      threshold: int = 10) -> bool:
    """
    Compare image hashes with adaptive thresholding.
    
    Args:
        hash1_tuple: First image hash tuple
        hash2_tuple: Second image hash tuple
        threshold: Similarity threshold
        
    Returns:
        bool: True if images are similar, False otherwise
    """
    if not hash1_tuple or not hash2_tuple:
        return False
        
    # Calculate differences for each hash type
    differences = [h1 - h2 for h1, h2 in zip(hash1_tuple, hash2_tuple)]
    
    # Weight the different hash types
    weights = [1.0, 1.2, 1.5, 0.8]  # Giving more weight to pHash
    weighted_diffs = [d * w for d, w in zip(differences, weights)]
    
    # Images are similar if the weighted average is below threshold
    return sum(weighted_diffs) / len(weighted_diffs) <= threshold

def process_images_batch(image_paths: List[str], batch_size: int = 50) -> Dict[str, Tuple[imagehash.ImageHash, ...]]:
    """
    Process images in batches to manage memory usage.
    
    Args:
        image_paths: List of image paths to process
        batch_size: Number of images to process at once
        
    Returns:
        Dict[str, Tuple[ImageHash, ...]]: Dictionary of image paths and their hashes
    """
    results = {}
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        for img_path in batch:
            hashes = calculate_image_hashes(img_path)
            if hashes:
                results[img_path] = hashes
    return results

def deduplicate_images(input_folder: str) -> Dict[str, int]:
    """
    Detect and handle duplicate images.
    
    Args:
        input_folder: Path to the folder containing images
        
    Returns:
        Dict[str, int]: Statistics about the deduplication process
    """
    stats = {"unique": 0, "duplicates": 0}
    
    try:
        # Get all PNG files
        image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
                      if f.lower().endswith('.png')]
        
        # Process images in batches
        image_hashes = process_images_batch(image_paths)
        
        # Find duplicate groups
        processed_files = set()
        duplicate_groups = []
        
        for filename, hash_tuple in image_hashes.items():
            if filename in processed_files:
                continue
                
            current_group = [filename]
            processed_files.add(filename)
            
            # Compare with remaining images
            for other_filename, other_hash in image_hashes.items():
                if other_filename not in processed_files and are_images_similar(hash_tuple, other_hash):
                    current_group.append(other_filename)
                    processed_files.add(other_filename)
            
            if len(current_group) > 1:
                duplicate_groups.append(current_group)
        
        # Handle duplicates
        discarded_folder = os.path.join(input_folder, "discarded")
        os.makedirs(discarded_folder, exist_ok=True)
        
        for group in duplicate_groups:
            # Keep the first file, move others to discarded
            stats["duplicates"] += len(group) - 1
            
            for duplicate in group[1:]:
                dst_path = os.path.join(discarded_folder, os.path.basename(duplicate))
                if not safe_file_move(duplicate, dst_path):
                    logger.warning(f"Failed to move duplicate: {duplicate}")
        
        stats["unique"] = len(image_hashes) - stats["duplicates"]
        
    except Exception as e:
        logger.error(f"Error during deduplication: {str(e)}")
        
    return stats

def main():
    try:
        # Create directory structure
        processed_dir = create_directory_structure()
        logger.info(f"Created processed directory: {processed_dir}")
        
        # Process all images
        base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        screenshots_dir = base_dir / "screenshots"
        
        if not screenshots_dir.exists():
            logger.error(f"Screenshots directory not found: {screenshots_dir}")
            return
        
        # Process images in batches
        stats = defaultdict(int)
        for filename in os.listdir(screenshots_dir):
            if filename.lower().endswith('.png'):
                source = screenshots_dir / filename
                
                if not validate_image(str(source)):
                    logger.warning(f"Skipping invalid image: {filename}")
                    continue
                
                # Determine appropriate destination
                if not has_dialogue_box(str(source)):
                    destination = Path(processed_dir) / "no_dialogue" / filename
                    stats["no_dialogue"] += 1
                else:
                    if filename.startswith("dialogue_box"):
                        destination = Path(processed_dir) / "box_dialogues" / filename
                        stats["box_dialogues"] += 1
                    else:
                        destination = Path(processed_dir) / "chara_dialogues" / filename
                        stats["chara_dialogues"] += 1
                
                if not safe_file_move(str(source), str(destination)):
                    logger.error(f"Failed to move {filename}")
        
        # Process duplicates in each relevant folder
        for folder in ["chara_dialogues", "box_dialogues"]:
            folder_path = os.path.join(processed_dir, folder)
            logger.info(f"\nProcessing {folder}:")
            dup_stats = deduplicate_images(folder_path)
            logger.info(f"Unique images: {dup_stats['unique']}")
            logger.info(f"Duplicates moved: {dup_stats['duplicates']}")
        
        logger.info("\nProcessing Summary:")
        logger.info(f"Character dialogues: {stats['chara_dialogues']}")
        logger.info(f"Box dialogues: {stats['box_dialogues']}")
        logger.info(f"Screenshots without dialogue: {stats['no_dialogue']}")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
# text utils (needs cleanup)
# merge Jsons out of image_to_text.py

import json
from typing import List, Union
from pathlib import Path
import pandas as pd

def merge_translation_jsons(file_paths: List[Union[str, Path]], output_path: str = None) -> list:
    """
    Merge multiple translation JSON files that contain a list of translation entries.
    
    Args:
        file_paths: List of paths to JSON files to merge
        output_path: Optional path to save merged JSON (if None, only returns merged list)
    
    Returns:
        list: Merged JSON data
    """
    merged_data = []
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            current_data = json.load(f)
            # Since current_data is a list, we can extend directly
            merged_data.extend(current_data)
    
    # Save to file if output path is provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    return merged_data

# Example usage:
if __name__ == "__main__":
    files = [r'C:\Users\andre\Desktop\OUTPUT_API\extracted_text_results_CHARA1.json',
            r"C:\Users\andre\Desktop\OUTPUT_API\extracted_text_results_DIALOGUES.json",
            r"C:\Users\andre\Desktop\OUTPUT_API\extracted_text_results_CHARA2.json",
            r"C:\Users\andre\Desktop\OUTPUT_API\extracted_text_results_CHARA3.json"]
    merged = merge_translation_jsons(files, r"C:\Users\andre\Desktop\OUTPUT_API\merged_translations.json")



## export to csv

def load_dialogue_data(file_path=r"C:\Users\andre\Desktop\OUTPUT_API\merged_translations.json"):
    # Read the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add sentence count column (using japanese_text as reference)
    df['sentence_count'] = df['japanese_text'].apply(len)
    
    return df

# Example usage
if __name__ == "__main__":
    # Load the data
    df = load_dialogue_data()


# deduplication and normalization of text to keep unique dialogues
import ast
import re

def normalize_japanese_punctuation(text):
    """
    Normalize text to use only Japanese punctuation:
    - Convert English ... to Japanese …
    - Convert English , to Japanese 、
    - Convert English . to Japanese 。
    - Convert English ! to Japanese ！
    - Convert English ? to Japanese ？
    - Remove any whitespace
    """
    # First evaluate the string representation of list
    if isinstance(text, str):
        try:
            text_list = ast.literal_eval(text)
        except:
            text_list = [text]
    else:
        text_list = text
    
    normalized_texts = []
    for t in text_list:
        # Remove whitespace
        t = ''.join(t.split())
        
        # Convert English punctuation to Japanese equivalents
        replacements = {
            r'\.{3,}': '…',  # Convert ... to …
            ',': '、',       # Convert , to 、
            r'(?<![。…])\.(?!\.)': '。',  # Convert single . to 。 (but not part of ...)
            '!': '！',       # Convert ! to ！
            r'\?': '？',     # Convert ? to ？
        }
        
        for eng, jap in replacements.items():
            t = re.sub(eng, jap, t)
        
        normalized_texts.append(t)
    
    return normalized_texts

def validate_japanese_punctuation(text):
    """
    Validate that only Japanese punctuation is present in the text.
    Returns (bool, str) tuple indicating if text is valid and explanation if not.
    """
    english_punct = re.findall(r'[,.!?]', text)
    if english_punct:
        return False, f"Contains English punctuation: {', '.join(english_punct)}"
    return True, ""

def create_joined_text(row):
    """Create the Joined_text column from japanese_text after normalization"""
    normalized = normalize_japanese_punctuation(row['japanese_text'])
    joined_text = ''.join(normalized)
    
    # Validate the normalized text
    is_valid, explanation = validate_japanese_punctuation(joined_text)
    if not is_valid:
        print(f"Warning: Row contains invalid punctuation: {explanation}")
        print(f"Original text: {row['japanese_text']}")
        print(f"Normalized text: {joined_text}")
    
    return joined_text

def find_duplicates(df):
    """
    Find both exact and substring duplicates within speaker groups.
    Returns two dataframes: unique entries and duplicates with explanations.
    """
    # Create Joined_text column
    df['Joined_text'] = df.apply(create_joined_text, axis=1)
    
    # Initialize results
    unique_entries = []
    duplicates = []
    
    # Process each speaker group separately
    for speaker, group in df.groupby('speaker'):
        # Sort by length of Joined_text (longer texts first)
        group = group.copy()
        group['text_length'] = group['Joined_text'].str.len()
        group = group.sort_values('text_length', ascending=False)
        
        # Track which entries we've processed
        processed_indices = set()
        
        for idx, row in group.iterrows():
            if idx in processed_indices:
                continue
                
            current_text = row['Joined_text']
            
            # Find potential duplicates
            potential_dupes = group[
                (group.index != idx) & 
                (~group.index.isin(processed_indices))
            ]
            
            for dupe_idx, dupe_row in potential_dupes.iterrows():
                dupe_text = dupe_row['Joined_text']
                
                # Check for exact duplicates first
                if current_text == dupe_text:
                    duplicates.append({
                        **dupe_row.to_dict(),
                        'duplicate_type': 'exact',
                        'matched_with': current_text,
                        'explanation': f'Exact duplicate of text from row {idx}'
                    })
                    processed_indices.add(dupe_idx)
                
                # Check for substring duplicates
                elif dupe_text in current_text:
                    duplicates.append({
                        **dupe_row.to_dict(),
                        'duplicate_type': 'substring',
                        'matched_with': current_text,
                        'explanation': f'Substring of longer text from row {idx}: {current_text}'
                    })
                    processed_indices.add(dupe_idx)
            
            if idx not in processed_indices:
                unique_entries.append(row.to_dict())
    
    # Convert results to dataframes
    unique_df = pd.DataFrame(unique_entries)
    duplicates_df = pd.DataFrame(duplicates)
    
    return unique_df, duplicates_df

def main(input_file):
    """Main processing function"""
    # Read input CSV
    df = pd.read_csv(input_file)
    
    # Process the data
    unique_df, duplicates_df = find_duplicates(df)
    
    # Save results
    unique_df.to_csv(r'C:\Users\andre\Desktop\unique_entries.csv', index=False)
    duplicates_df.to_csv(r'C:\Users\andre\Desktop\duplicates.csv', index=False)
    
    print(f"\nProcessing complete:")
    print(f"Found {len(unique_df)} unique entries")
    print(f"Found {len(duplicates_df)} duplicates")
    
    return unique_df, duplicates_df

if __name__ == "__main__":
    main(r"C:\Users\andre\Desktop\asd.csv")
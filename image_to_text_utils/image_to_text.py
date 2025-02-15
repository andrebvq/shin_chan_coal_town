import os
import base64
import anthropic
import asyncio
from pathlib import Path
import logging
import json
from dataclasses import dataclass
from datetime import datetime
import mimetypes
from typing import List

@dataclass
class ExtractionResult:
    image_name: str
    speaker: str
    japanese_text: List[str]  # Changed to List to store separate sentences
    romaji: List[str]         # Changed to List to match japanese_text
    translation: List[str]    # Changed to List to match japanese_text
    translation_notes: str
    ocr_confidence: float
    translation_confidence: float
    
    def to_dict(self):
        return {
            'image_name': self.image_name,
            'speaker': self.speaker,
            'japanese_text': self.japanese_text,
            'romaji': self.romaji,
            'translation': self.translation,
            'translation_notes': self.translation_notes,
            'ocr_confidence': self.ocr_confidence,
            'translation_confidence': self.translation_confidence
        }

class ClaudeTextExtractor:
    def __init__(self, api_key: str, max_concurrent: int = 2):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.max_concurrent = max_concurrent
        self.setup_logging()

    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'text_extraction_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logging initialized")

    def get_mime_type(self, file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or 'image/jpeg'

    def encode_image(self, image_path: str) -> tuple[str, str]:
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                encoded_data = base64.b64encode(image_data).decode('utf-8')
                mime_type = self.get_mime_type(image_path)
                self.logger.info(f"Successfully encoded image: {image_path}")
                return encoded_data, mime_type
        except Exception as e:
            self.logger.error(f"Error encoding image {image_path}: {str(e)}")
            raise

    async def extract_text_from_image(self, image_path: str) -> ExtractionResult:
        image_name = Path(image_path).name
        speaker = image_name.split('_')[0] if '_' in image_name else image_name
        
        try:
            image_data, mime_type = self.encode_image(image_path)
            
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": "Return a JSON object with these fields only:"
                                    "- japanese_text: array of Japanese text sentences in the image (exclude text in brown boxes)"
                                    "- romaji: array of romanized versions of each Japanese sentence"
                                    "- translation: array of English translations for each sentence"
                                    "- translation_notes: any notes or context about the translation"
                                    "- ocr_confidence: confidence score (1 if text is correctly inferred)"
                                    "- translation_confidence: confidence score (1 if translation is accurate)"
                                    "\nMake sure japanese_text, romaji, and translation arrays have matching lengths"
                        }
                    ]
                }]
            )
            
            try:
                response_data = json.loads(message.content[0].text)
                
                # Ensure we have lists, even if Claude returns single strings
                def ensure_list(value):
                    if isinstance(value, str):
                        return [value]
                    return value if value else []

                japanese_text = ensure_list(response_data['japanese_text'])
                romaji = ensure_list(response_data['romaji'])
                translation = ensure_list(response_data['translation'])

                # Validate array lengths match
                if not (len(japanese_text) == len(romaji) == len(translation)):
                    self.logger.warning(f"Mismatched array lengths for {image_name}. Padding shorter arrays.")
                    max_len = max(len(japanese_text), len(romaji), len(translation))
                    japanese_text.extend([''] * (max_len - len(japanese_text)))
                    romaji.extend([''] * (max_len - len(romaji)))
                    translation.extend([''] * (max_len - len(translation)))

                return ExtractionResult(
                    image_name=image_name,
                    speaker=speaker,
                    japanese_text=japanese_text,
                    romaji=romaji,
                    translation=translation,
                    translation_notes=response_data.get('translation_notes', ''),
                    ocr_confidence=float(response_data['ocr_confidence']),
                    translation_confidence=float(response_data['translation_confidence'])
                )
                
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parse error: {str(e)}")
                self.logger.error(f"Raw response: {message.content[0].text}")
                raise
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            raise

    async def process_directory(self, directory_path: str, output_file: str) -> List[ExtractionResult]:
        # Rest of the process_directory method remains the same
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        image_paths = [
            str(p) for p in Path(directory_path).glob("**/*") 
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif'}
        ]
        
        self.logger.info(f"Found {len(image_paths)} images to process")
        
        if not image_paths:
            return []

        results = []
        for i in range(0, len(image_paths), self.max_concurrent):
            batch = image_paths[i:i + self.max_concurrent]
            batch_results = await asyncio.gather(
                *[self.extract_text_from_image(img) for img in batch],
                return_exceptions=True
            )
            
            for j, result in enumerate(batch_results):
                if not isinstance(result, Exception):
                    results.append(result)
                else:
                    self.logger.error(f"Failed to process {image_paths[i+j]}: {str(result)}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Processed {i + len(batch)}/{len(image_paths)} images")
            
            if i + self.max_concurrent < len(image_paths):
                await asyncio.sleep(2)
        
        return results

async def main():
    api_key = "Claude-API-Key"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_directory = os.path.join(current_dir, "screenshots_to_text")
    output_file = os.path.join(current_dir, "extracted_text_results.json")
    
    extractor = ClaudeTextExtractor(api_key=api_key, max_concurrent=2)
    
    try:
        results = await extractor.process_directory(input_directory, output_file)
        print(f"Successfully processed {len(results)} images")
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
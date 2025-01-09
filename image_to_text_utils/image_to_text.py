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
    first_word: str
    japanese_text: str
    translation: str
    ocr_confidence: float
    translation_confidence: float
    
    def to_dict(self):
        return {
            'image_name': self.image_name,
            'first_word': self.first_word,
            'japanese_text': self.japanese_text.replace('\n', ' ').strip(),
            'translation': self.translation.replace('\n', ' ').strip(),
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
        first_word = image_name.split('_')[0] if '_' in image_name else image_name
        
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
                                    "- japanese_text: the Japanese text in the image (exclude text in brown boxes)"
                                    "- translation: English translation"
                                    "- ocr_confidence: confidence score (1 if text is correctly inferred)"
                                    "- translation_confidence: confidence score (1 if translation is accurate)"
                        }
                    ]
                }]
            )
            
            try:
                response_data = json.loads(message.content[0].text)
                return ExtractionResult(
                    image_name=image_name,
                    first_word=first_word,
                    japanese_text=response_data['japanese_text'].replace('\n', ' ').strip(),
                    translation=response_data['translation'].replace('\n', ' ').strip(),
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
    api_key = "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"
    
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
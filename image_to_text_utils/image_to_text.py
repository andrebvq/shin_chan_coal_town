import os
import base64
import anthropic
import asyncio
from pathlib import Path
import logging
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import mimetypes

@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    
    def __add__(self, other):
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens
        )

@dataclass
class ExtractionResult:
    image_path: str
    success: bool
    extraction: str = None
    error: str = None
    token_usage: TokenUsage = None
    
    def to_dict(self):
        return {k: v if not isinstance(v, TokenUsage) else asdict(v) 
                for k, v in asdict(self).items()}

class ClaudeTextExtractor:
    def __init__(self, api_key: str, max_concurrent: int = 3):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.max_concurrent = max_concurrent
        self.total_token_usage = TokenUsage(0, 0)
        self.setup_logging()

    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f'text_extraction_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_mime_type(self, file_path: str) -> str:
        """
        Get the correct MIME type for the image file
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type is None:
            # Default to JPEG if we can't determine the type
            mime_type = 'image/jpeg'
        return mime_type

    def encode_image(self, image_path: str) -> tuple[str, str]:
        """
        Encode an image file to base64 and return its MIME type
        Returns: (base64_string, mime_type)
        """
        with open(image_path, "rb") as image_file:
            # Read the file in binary mode
            image_data = image_file.read()
            # Encode to base64
            encoded_data = base64.b64encode(image_data).decode('utf-8')
            # Get the correct MIME type
            mime_type = self.get_mime_type(image_path)
            
            self.logger.info(f"Encoded image size: {len(encoded_data)} bytes, MIME type: {mime_type}")
            return encoded_data, mime_type

    async def extract_text_from_image(self, image_path: str) -> ExtractionResult:
        try:
            image_data, mime_type = self.encode_image(image_path)
            
            # Add a small delay between requests to avoid rate limiting
            await asyncio.sleep(2.5)
            
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
                            "text": "Please extract all text from this image and return it as a JSON object with two fields: 'raw_text' for the exact text as it appears, and 'translation' for the English translation if the text is in Japanese. If the text is in English, the translation field should be null."
                        }
                    ]
                }]
            )
            
            # Extract token usage if available
            token_usage = None
            if hasattr(message, 'usage'):
                token_usage = TokenUsage(
                    input_tokens=message.usage.input_tokens,
                    output_tokens=message.usage.output_tokens
                )
            
            if token_usage:
                self.total_token_usage += token_usage
            
            self.logger.info(
                f"Successfully processed {image_path}\n"
                f"Token usage: Input={token_usage.input_tokens}, "
                f"Output={token_usage.output_tokens}"
            )
            
            return ExtractionResult(
                image_path=image_path,
                success=True,
                extraction=message.content[0].text,
                token_usage=token_usage
            )
            
        except anthropic.NotFoundError as e:
            self.logger.error(f"Model not found error for {image_path}: {str(e)}")
            return ExtractionResult(
                image_path=image_path,
                success=False,
                error=f"Model not found: {str(e)}"
            )
        except anthropic.APIError as e:
            self.logger.error(f"API Error for {image_path}: {str(e)}")
            return ExtractionResult(
                image_path=image_path,
                success=False,
                error=f"API Error: {str(e)}"
            )
        except Exception as e:
            self.logger.error(f"Unexpected error for {image_path}: {str(e)}")
            return ExtractionResult(
                image_path=image_path,
                success=False,
                error=f"Unexpected error: {str(e)}"
            )

    async def process_directory(self, directory_path: str, output_file: str = None):
        # Initialize mimetypes
        mimetypes.init()
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"extracted_text_{timestamp}.json"
        
        image_paths = [
            str(p) for p in Path(directory_path).glob("**/*") 
            if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif'}
        ]
        
        self.logger.info(f"Found {len(image_paths)} images to process")
        
        results = []
        for i in range(0, len(image_paths), self.max_concurrent):
            batch = image_paths[i:i + self.max_concurrent]
            batch_results = await asyncio.gather(
                *[self.extract_text_from_image(img) for img in batch]
            )
            results.extend(batch_results)
            
            self.logger.info(
                f"Processed {i + len(batch)}/{len(image_paths)} images. "
                f"Running total tokens: "
                f"Input={self.total_token_usage.input_tokens}, "
                f"Output={self.total_token_usage.output_tokens}"
            )
            
            # Save intermediate results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
        
        successful = sum(1 for r in results if r.success)
        
        summary = f"""
        Processing Summary
        -----------------
        Total images processed: {len(results)}
        Successful extractions: {successful}
        Failed extractions: {len(results) - successful}
        
        Token Usage
        -----------
        Input tokens: {self.total_token_usage.input_tokens:,}
        Output tokens: {self.total_token_usage.output_tokens:,}
        Total tokens: {self.total_token_usage.input_tokens + self.total_token_usage.output_tokens:,}
        
        Results saved to: {output_file}
        """
        
        self.logger.info(summary)
        return results

async def main():
    """
    Main function to run the text extraction
    """
    api_key = "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"
    if not api_key:
        raise ValueError("Please set the ANTHROPIC_API_KEY environment variable")
        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_directory = os.path.join(current_dir, "screenshots_deduped")
    
    extractor = ClaudeTextExtractor(api_key=api_key)
    await extractor.process_directory(input_directory)

if __name__ == "__main__":
    asyncio.run(main())
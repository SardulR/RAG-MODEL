
import base64
from PIL import Image
from io import BytesIO
from typing import Optional
from langchain.schema import Document
import logging
from langchain_openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageHandler:
    def __init__(self):
        """Initialize OpenAI client for image processing"""
        try:
            logger.info("Initializing image handler...")
            self.llm = OpenAI(temperature=0)
            logger.info("Successfully initialized image handler")
        except Exception as e:
            logger.error(f"Error initializing image handler: {e}")
            raise

    def process_image(self, image_path: str) -> Optional[Document]:
        """Process image and return document with description"""
        try:
            # Load and process image
            logger.info(f"Processing image: {image_path}")
            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize if image is too large
                if max(img.size) > 2048:
                    ratio = 2048 / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)

                # Convert image to base64 for API
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                # Get image description using OpenAI
                response = self.llm.invoke(
                    [
                        {
                            "type": "text",
                            "text": "Describe this image in detail:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_str}"
                            }
                        }
                    ]
                )

                description = response.content if hasattr(response, 'content') else str(response)

                # Create document with description
                doc = Document(
                    page_content=description,
                    metadata={
                        "source": image_path,
                        "category": "Image",
                        "modality": "image",
                        "size": img.size
                    }
                )
                logger.info(f"Successfully processed image: {image_path}")
                return doc

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None

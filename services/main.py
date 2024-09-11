import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from services.detector import download_images, process_segmentation

# Load environment variables from .env file
load_dotenv()

MEDIA_URL = os.getenv("MEDIA_URL")

app = FastAPI()


class ImageURLs(BaseModel):
    urls: list[str]


@app.post("/process_images/")
async def process_images(request: ImageURLs):
    image_urls = request.urls

    try:
        image_paths = download_images(image_urls)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading images: {e}")

    if not image_paths:
        return {"error": "No valid images to process"}

    try:
        processed_image_urls, detected_classes, terminal_output = process_segmentation(
            image_paths
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {e}")

    return {
        "results": {
            "detected": detected_classes,
            "processed_image_urls": processed_image_urls,
            "terminal_output": terminal_output,
        }
    }

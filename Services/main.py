from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from Services.detector import download_images, process_segmentation
import os
import tempfile

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost:5173/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Serve static files from the result_images directory
app.mount(
    "/media/result_images",
    StaticFiles(
        directory="/home/franzy/CancerCellDetection/media/images/result_images"
    ),
    name="result_images",
)


# Define request model
class ImageURLs(BaseModel):
    urls: list[str]


@app.post("/process_images/")
async def process_images(request: ImageURLs):
    image_urls = request.urls
    temp_dir = os.path.join(tempfile.gettempdir(), "user", "temp")
    result_folder = "/home/franzy/CancerCellDetection/media/images/result_images/"

    # Download images
    try:
        image_paths = download_images(image_urls, temp_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading images: {e}")

    if not image_paths:
        return {"error": "No valid images to process"}

    # Process the segmentation
    try:
        processed_image_urls, detected_classes = process_segmentation(
            image_paths, result_folder
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {e}")

    # Return the API response
    return {
        "results": {
            "detected": detected_classes,
            "processed_image_urls": processed_image_urls,
        }
    }

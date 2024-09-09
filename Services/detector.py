import os
import re
import requests
import subprocess
import tempfile
from typing import List, Tuple
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def download_images(image_urls: List[str], temp_dir: str) -> List[str]:
    """Download images from URLs and save them in a temporary directory."""
    os.makedirs(temp_dir, exist_ok=True)
    image_paths = []

    for url in image_urls:
        image_name = os.path.basename(url)
        image_path = os.path.join(temp_dir, image_name)

        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(image_path, "wb") as f:
                f.write(response.content)
            image_paths.append(image_path)
            logging.info(f"Downloaded {url} to {image_path}")
        except requests.RequestException as e:
            logging.error(f"Failed to download {url}: {e}")

    return image_paths


def process_segmentation(
    image_paths: List[str], result_folder: str
) -> Tuple[List[str], dict]:
    """Run the segmentation model on the images and return processed results."""
    os.makedirs(result_folder, exist_ok=True)

    command = [
        "source /home/franzy/Envs/Cell_Detection/bin/activate, python detect.py"
        + " ".join(image_paths),
    ]

    # Start the subprocess to run the command
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False
    )
    stdout, stderr = process.communicate()

    # Extract processed image URLs
    processed_image_urls = [
        f"http://localhost:8000/media/result_images/{os.path.basename(path).replace('.jpg', '_overlayed.jpg')}"
        for path in image_paths
    ]

    # Extract detection information from stderr
    detected_classes = {}
    detection_results = stdout.decode("utf-8") + stderr.decode("utf-8")
    for match in re.findall(r"(\w+):\s*(\d+)\s*pixels", detection_results):
        class_name, pixel_count = match
        detected_classes[class_name] = int(pixel_count)

    return processed_image_urls, detected_classes

import os
import subprocess
import logging
import tempfile
from typing import List, Tuple
import re
from dotenv import load_dotenv
import requests
from requests.exceptions import RequestException

# Load environment variables from .env file
load_dotenv()

RESULT_FOLDER = os.getenv("RESULT_FOLDER")
VENV_PATH = os.getenv("VENV_PATH")
TEMP_DIR = os.getenv("TEMP_DIR", tempfile.gettempdir())

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def download_images(image_urls: List[str]) -> List[str]:
    """Download images from URLs and save them in a temporary directory."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    image_paths = []

    for url in image_urls:
        image_name = os.path.basename(url)
        image_path = os.path.join(TEMP_DIR, image_name)

        try:
            # Normalize URL and log it
            url = url.replace("\\", "/")
            print(f"Downloading image from: {url}")

            # Make the request with a user-agent
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, allow_redirects=True)

            # Log response details
            print(f"Response Status Code: {response.status_code}")
            print(f"Response Final URL: {response.url}")

            response.raise_for_status()

            # Write content to file
            with open(image_path, "wb") as f:
                f.write(response.content)
            image_paths.append(image_path)
            logging.info(f"Downloaded {url} to {image_path}")
        except RequestException as e:  # Updated to requests.RequestException
            logging.error(f"Failed to download {url}: {e}")

    return image_paths

def process_segmentation(image_paths: List[str]) -> Tuple[List[str], dict, str]:
    """Run the segmentation model on the images and return processed results."""
    if not VENV_PATH:
        raise EnvironmentError("Virtual environment path (VENV_PATH) is not set")

    python_executable = os.path.join(VENV_PATH)
    if not os.path.isfile(python_executable):
        raise FileNotFoundError(f"Python executable not found at {python_executable}")

    command = [python_executable, "detect.py"] + image_paths
    logging.info(f"Running command: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )

        stdout = result.stdout
        stderr = result.stderr

        logging.info("Segmentation process completed successfully.")
        logging.info(f"Output:\n{stdout}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Segmentation process failed with error: {e.stderr}")
        raise RuntimeError(f"Segmentation process failed: {e.stderr}")
    except Exception as e:
        logging.error(f"Unexpected error during segmentation: {str(e)}")
        raise RuntimeError(f"Unexpected error during segmentation: {str(e)}")

    processed_image_urls = [
        f"media/images/result_images/{os.path.basename(path).replace('.jpg', '_overlayed.jpg')}"

        for path in image_paths
    ]

    detected_classes = {}
    detection_results = stdout + stderr
    logging.info(f"Segmentation output:\n{detection_results}")

    for match in re.findall(r"(\w+):\s*(\d+)\s*pixels", detection_results):
        class_name, pixel_count = match
        detected_classes[class_name] = int(pixel_count)

    return processed_image_urls, detected_classes, detection_results

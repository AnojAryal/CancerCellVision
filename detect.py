import sys
import torch
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
import numpy as np
import cv2
from collections import Counter
import os

# Define and load the model
model = smp.Unet(
    encoder_name="resnet34",  # Same encoder used during training
    encoder_weights=None,  # No pre-trained weights, as we are loading our own
    in_channels=3,  # Number of input channels
    classes=6,  # Number of output classes
)

device = torch.device("cuda:0")

# Load model weights
checkpoint = torch.load("segmentation_model.pth", map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()  # Set the model to evaluation mode

# Define the transformation pipeline (same as used during training)
transform = A.Compose(
    [
        A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST),
        A.Normalize(
            mean=[0.17066666, 0.17066666, 0.17066666],
            std=[0.4131182, 0.3946274, 0.4131182],
        ),
        ToTensorV2(),
    ]
)

# Load class names
class_names = np.array(
    [
        "Neoplastic cells",
        "Inflammatory",
        "Connective/Soft tissue cells",
        "Dead Cells",
        "Epithelial",
        "Background",
    ]
)


def preprocess_image(image_path):
    """Load and preprocess the image."""
    image = np.array(Image.open(image_path).convert("RGB"))
    transformed = transform(image=image)
    return transformed["image"].unsqueeze(0), image


def predict(image_path):
    """Make a prediction using the model."""
    input_image, original_image = preprocess_image(image_path)
    input_image = input_image.to(device)

    with torch.no_grad():
        output = model(input_image)

    predicted_mask = torch.argmax(output, dim=1).cpu().numpy()[0]
    return predicted_mask, original_image


def overlay_mask_on_image(original_image, predicted_mask, alpha=0.5):
    """Overlay the predicted mask on the original image."""
    predicted_mask_resized = cv2.resize(
        predicted_mask,
        (original_image.shape[1], original_image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    mask_colored = cv2.applyColorMap(
        (predicted_mask_resized * (255 // predicted_mask_resized.max())).astype(
            np.uint8
        ),
        cv2.COLORMAP_JET,
    )
    blended_image = cv2.addWeighted(original_image, 1 - alpha, mask_colored, alpha, 0)
    return Image.fromarray(blended_image)


def analyze_mask(predicted_mask):
    """Analyze the mask to find and count detected classes."""
    flat_mask = predicted_mask.flatten()
    class_counts = Counter(flat_mask)
    detected_classes = {}
    for class_id, count in class_counts.items():
        if class_id < len(class_names):
            detected_classes[class_names[class_id]] = count
    return detected_classes


def process_images(image_paths):
    """Process each image and output results."""
    result_dir = "/home/franzy/CancerCellDetection/media/images/result_images/"

    # Ensure the result directory exists
    if not os.path.exists(result_dir):
        try:
            os.makedirs(result_dir)
            print(f"Directory created at: {result_dir}")
        except Exception as e:
            print(f"Error creating directory: {e}")
            return

    for image_path in image_paths:
        try:
            # Assuming predict() returns a predicted mask and the original image
            predicted_mask, original_image = predict(image_path)

            if predicted_mask is None or original_image is None:
                print(f"Skipping invalid image: {image_path}")
                continue

            # Preserve the original file name and extension
            base_name = os.path.basename(image_path)
            result_image_path = os.path.join(result_dir, base_name)

            # Overlay the mask on the image and save it
            overlaid_image = overlay_mask_on_image(original_image, predicted_mask)

            # Check if the image is valid before saving
            if isinstance(overlaid_image, Image.Image):
                overlaid_image.save(result_image_path)
                print(f"Saved overlayed image to {result_image_path}")
            else:
                print(f"Error: Overlayed image is not valid for {image_path}")

            # Analyze the mask and print results
            detected_classes = analyze_mask(predicted_mask)
            result_text = f"{base_name}: " + ", ".join(
                [f"{cls}: {count} pixels" for cls, count in detected_classes.items()]
            )
            print(result_text)

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Process and analyze images.")
    parser.add_argument(
        "image_paths",
        metavar="image_path",
        type=str,
        nargs="+",
        help="Paths to images to process (comma-separated or space-separated)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_images(args.image_paths)

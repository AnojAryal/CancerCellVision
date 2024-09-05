import sys
import torch
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
import numpy as np
 # OpenCV for color mapping
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model weights
checkpoint = torch.load(
    "segmentation_model.pth", map_location=device, weights_only=True
)
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

# Load class names from types.npy
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
    # Return both transformed image and original image
    return transformed["image"].unsqueeze(0), image


def predict(image_path):
    """Make a prediction using the model."""
    input_image, original_image = preprocess_image(image_path)
    input_image = input_image.to(device)

    # Make a prediction
    with torch.no_grad():
        output = model(input_image)

    # Get the predicted mask (argmax across classes)
    predicted_mask = torch.argmax(output, dim=1).cpu().numpy()[0]

    return predicted_mask, original_image


def overlay_mask_on_image(original_image, predicted_mask, alpha=0.5):
    """Overlay the predicted mask on the original image."""
    # Resize predicted mask to match the original image size
    predicted_mask_resized = cv2.resize(
        predicted_mask,
        (original_image.shape[1], original_image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    # Convert predicted mask to color
    mask_colored = cv2.applyColorMap(
        (predicted_mask_resized * (255 // predicted_mask_resized.max())).astype(
            np.uint8
        ),
        cv2.COLORMAP_JET,
    )

    # Blend original image with colored mask
    blended_image = cv2.addWeighted(original_image, 1 - alpha, mask_colored, alpha, 0)

    # Convert back to PIL format for visualization
    blended_image_pil = Image.fromarray(blended_image)

    return blended_image_pil


def analyze_mask(predicted_mask):
    """Analyze the mask to find and count detected classes."""
    # Flatten the mask array and count occurrences of each class
    flat_mask = predicted_mask.flatten()
    class_counts = Counter(flat_mask)

    # Prepare detected classes and their counts
    detected_classes = {}
    for class_id, count in class_counts.items():
        if class_id < len(class_names):
            detected_classes[class_names[class_id]] = count

    return detected_classes


def process_images(image_paths):
    """Process each image and output results."""
    for image_path in image_paths:
        predicted_mask, original_image = predict(image_path)

        # Save the overlaid image
        result_image_path = os.path.join(
            "Results", os.path.basename(image_path).replace(".jpg", "_overlayed.jpg")
        )
        overlaid_image = overlay_mask_on_image(original_image, predicted_mask)
        overlaid_image.save(result_image_path)

        # Analyze the mask and print results
        detected_classes = analyze_mask(predicted_mask)
        result_text = f"{os.path.basename(image_path)}: " + ", ".join(
            [f"{cls}: {count} pixels" for cls, count in detected_classes.items()]
        )
        print(result_text)


if __name__ == "__main__":
    # Check for image paths from the command line
    if len(sys.argv) < 2:
        print("Usage: python detect.py <image_path1,image_path2,...>")
        sys.exit(1)

    # Get image paths from command line
    image_paths = sys.argv[1].split(",")

    # Create Results folder if not exists
    os.makedirs("Results", exist_ok=True)

    # Process images
    process_images(image_paths)

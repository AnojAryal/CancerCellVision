import torch
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV for color mapping
from collections import Counter

# Define and load the model
model = smp.Unet(
    encoder_name="resnet34",        # Same encoder used during training
    encoder_weights=None,           # No pre-trained weights, as we are loading our own
    in_channels=3,                  # Number of input channels
    classes=6                       # Number of output classes
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model weights with weights_only=True
model.load_state_dict(torch.load("segmentation_model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Define the transformation pipeline (same as used during training)
transform = A.Compose([
    A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST),
    A.Normalize(mean=[0.17066666, 0.17066666, 0.17066666], std=[0.4131182, 0.3946274, 0.4131182]),
    ToTensorV2()
])

# Load class names from types.npy
class_names = np.array([
    "Neoplastic cells",
    "Inflammatory",
    "Connective/Soft tissue cells",
    "Dead Cells",
    "Epithelial",
    "Background"
])

def preprocess_image(image_path):
    """Load and preprocess the image."""
    image = np.array(Image.open(image_path).convert("RGB"))
    transformed = transform(image=image)
    return transformed["image"].unsqueeze(0), image  # Return both transformed image and original image

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
    predicted_mask_resized = cv2.resize(predicted_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert predicted mask to color
    mask_colored = cv2.applyColorMap((predicted_mask_resized * (255 // predicted_mask_resized.max())).astype(np.uint8), cv2.COLORMAP_JET)

    # Blend original image with colored mask
    blended_image = cv2.addWeighted(original_image, 1 - alpha, mask_colored, alpha, 0)
    
    # Convert back to PIL format for visualization
    blended_image_pil = Image.fromarray(blended_image)

    return blended_image_pil

def visualize_prediction(image_path, alpha=0.5):
    """Display the original image and the mask overlay side by side."""
    predicted_mask, original_image = predict(image_path)
    overlaid_image = overlay_mask_on_image(original_image, predicted_mask, alpha=alpha)
    
    # Convert original image to PIL for consistency
    original_image_pil = Image.fromarray(original_image)
    
    # Display images side by side
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_pil)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(overlaid_image)
    plt.title("Image with Mask Overlay")
    plt.axis('off')
    
    plt.show()

def analyze_mask(predicted_mask):
    """Analyze the mask to find and count detected classes."""
    # Flatten the mask array and count occurrences of each class
    flat_mask = predicted_mask.flatten()
    class_counts = Counter(flat_mask)
    
    # Print detected classes and their counts
    detected_classes = {}
    for class_id, count in class_counts.items():
        if class_id < len(class_names):
            detected_classes[class_names[class_id]] = count
    
    return detected_classes

# Use the script to predict, visualize the result, and analyze detected classes
if __name__ == "__main__":
    image_path = r"data\histology_1.jpg"  # Replace with the actual path to your image
    visualize_prediction(image_path, alpha=0.5)
    
    # Obtain and print detected classes and their counts
    predicted_mask, _ = predict(image_path)
    detected_classes = analyze_mask(predicted_mask)
    print("Detected Classes and Counts:")
    for class_name, count in detected_classes.items():
        print(f"{class_name}: {count} pixels")

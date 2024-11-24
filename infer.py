import argparse
import torch
import cv2
import os
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import segmentation_models_pytorch as smp

# Color mapping for segmentation
color_dict = {0: (0, 0, 0),  # Background
              1: (255, 0, 0),  # Class 1
              2: (0, 255, 0)}  # Class 2

# Function to map mask to RGB using the color dictionary
def mask_to_rgb(mask, color_dict):
    output = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for k, color in color_dict.items():
        output[mask == k] = color
    return output

def main(image_path):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained model
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,  # No pre-trained weights as we're loading our checkpoint
        in_channels=3,
        classes=3
    )
    model.to(device)

    # Load the checkpoint
    checkpoint = torch.load('colorization_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Transform for input image
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Read and preprocess the input image
    ori_img = cv2.imread(image_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_h, ori_w = ori_img.shape[:2]

    resized_img = cv2.resize(ori_img, (256, 256))  # Resize to match model input size
    transformed = transform(image=resized_img)
    input_img = transformed['image'].unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output_mask = model(input_img).squeeze(0).cpu().numpy().transpose(1, 2, 0)

    # Convert the output to RGB mask
    mask = cv2.resize(output_mask, (ori_w, ori_h))
    mask = np.argmax(mask, axis=2)
    mask_rgb = mask_to_rgb(mask, color_dict)

    # Save the result
    output_path = "segmented_output.png"
    mask_rgb_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, mask_rgb_bgr)
    print(f"Segmented image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment an image using a pre-trained model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: The file {args.image_path} does not exist.")
    else:
        main(args.image_path)

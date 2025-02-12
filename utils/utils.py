import time

import torch
from torch import nn
import argparse
import json
from torchvision import transforms
from PIL import Image
import numpy as np

TRAIN = "train"
VALIDATE = "validate"
TEST = "test"
LINEAR_SIZE = 512


def input_args():
    parser = argparse.ArgumentParser(description="Train network on a dataset")
    parser.add_argument("--data_dir", type=str, default="flowers", help="Path to the dataset directory")
    parser.add_argument("--save_dir", type=str, default="checkpoints/", help="Directory to save model checkpoints")
    parser.add_argument("--arch", choices=["vgg16_bn", "densenet121", "resnet50", "alexnet"], default="vgg16_bn",
                        help="Model architecture")
    parser.add_argument("--hidden_units", type=int, default=4096, help="Number of hidden units in the classifier")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training if available")

    return parser.parse_args()


def get_args():
    parser = argparse.ArgumentParser(description='Predict class of input image')

    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default='flower_classifier_checkpoint.pth',
                        help='Path to the trained model checkpoint file')
    parser.add_argument('--gpu', action='store_true', help="Enable GPU for inference")
    parser.add_argument('--top_k', type=int, default=5, help="Number of top likely classes to return")
    parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json',
                        help='Path to JSON file mapping categories to names')

    return parser.parse_args()


def get_labels(labels_path="cat_to_name.json"):
    with open(labels_path, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name


def save_checkpoint(model, hidden_units, output_units, save_dir, arch, class_to_index):
    """Saves the trained model checkpoint."""
    checkpoint = {
        'architecture': arch,
        'input_size': model.classifier[0].in_features,
        'hidden_units': hidden_units,
        'output_size': output_units,
        'state_dict': model.state_dict(),
        'class_to_index': class_to_index,
        'classifier': model.classifier
    }

    # Define save path
    filename = f"{arch}_checkpoint{time.time()}.pth"
    save_path = f"{save_dir}/{filename}" if save_dir else filename

    # Save the checkpoint
    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved at: {save_path}")


def load_model_checkpoint(filepath, gpu):
    """Loads a model checkpoint and rebuilds the model."""

    # Set map_location based on the device
    map_location = 'cuda' if gpu else 'cpu'

    # Load the checkpoint
    checkpoint = torch.load(filepath, map_location=map_location)

    # Return model architecture details and the state_dict
    return checkpoint['architecture'], checkpoint['input_size'], checkpoint['output_size'], checkpoint['hidden_units'], \
        checkpoint['state_dict'], checkpoint['class_to_index'], checkpoint['classifier']


def process_image(image_path):
    """
    Preprocesses an image for use in a PyTorch model.
    The image is resized, center-cropped, converted to a tensor, and normalized.
    """
    # Open the image and convert it to RGB
    image = Image.open(image_path).convert('RGB')

    # Resize image while maintaining aspect ratio
    image.thumbnail((256, 256))
    width, height = image.size  # Get the current image dimensions

    # Compute coordinates for center crop
    crop_width, crop_height = 224, 224
    left = (width - crop_width) / 2
    top = (height - crop_height) / 2
    right = (width + crop_width) / 2
    bottom = (height + crop_height) / 2

    # Apply center crop
    image = image.crop((left, top, right, bottom))

    # Define transformations: Convert to tensor and normalize
    transform_to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Apply transformations
    tensor_image = normalize(transform_to_tensor(image))

    # Convert tensor to numpy array
    processed_image_np = np.array(tensor_image)

    return processed_image_np

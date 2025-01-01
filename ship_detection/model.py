#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: model.py
Author: Adam Muhtar <adam.b.muhtar@gmail.com>
Description: PyTorch model classes and utility functions for ship classification.
"""

# Standard library imports
import logging
from pathlib import Path
from typing import List, Literal, Tuple, Union

# Third party imports
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms


class ShipClassifier2ConvNet(nn.Module):
    """Convolutional Neural Network for ship classification."""
    def __init__(self, dropout: float = 0.2) -> None:
        """
        Initialise the ShipClassifier2ConvNet class. The model consists of four
        convolutional layers followed by two fully connected layers.
        
        The convolutional layers are as follows:
        - Conv1: 3 input channels (RGB), 32 feature maps, 3x3 kernel, stride 1,
            padding 1
        - Conv2: 32 input channels, 64 feature maps, 3x3 kernel, stride 1,
            padding 1
        
        The fully connected layers are as follows:
        - Linear1: 64 * 64 * 64 input features, 512 output features
        - Linear2: 512 input features, 2 output features (ship or no ship)
        
        The ReLU activation function is used after each convolutional and fully
        connected layer, except for the output layer. Max pooling is applied
        after each convolutional layer.
        
        Batch normalisation is applied after each convolutional layer to
        normalise the activations and improve training speed.
        
        Dropout is applied after the first fully connected layer to prevent
        overfitting.
        
        Args:
            dropout (`float`): Dropout probability for the fully connected layer
                to prevent overfitting. Default is 0.2.
        """
        super(ShipClassifier2ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional layer
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=(64*64*64), out_features=512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=2)  # Output 2 classes (ship or no ship)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class ShipClassifier4ConvNet(nn.Module):
    """Convolutional Neural Network for ship classification."""
    def __init__(self, dropout: float = 0.2) -> None:
        """
        Initialise the ShipClassifier4ConvNet class. The model consists of four
        convolutional layers followed by two fully connected layers.
        
        The convolutional layers are as follows:
        - Conv1: 3 input channels (RGB), 32 feature maps, 3x3 kernel, stride 1,
            padding 1
        - Conv2: 32 input channels, 64 feature maps, 3x3 kernel, stride 1,
            padding 1
        - Conv3: 64 input channels, 128 feature maps, 3x3 kernel, stride 1,
            padding 1
        - Conv4: 128 input channels, 256 feature maps, 3x3 kernel, stride 1,
            padding 1
        
        The fully connected layers are as follows:
        - Linear1: 256 * 32 * 32 input features, 512 output features
        - Linear2: 512 input features, 2 output features (ship or no ship)
        
        The ReLU activation function is used after each convolutional and fully
        connected layer, except for the output layer. Max pooling is applied
        after each convolutional layer.
        
        Batch normalisation is applied after each convolutional layer to
        normalise the activations and improve training speed.
        
        Dropout is applied after the first fully connected layer to prevent
        overfitting.
        
        Args:
            dropout (`float`): Dropout probability for the fully connected layer
                to prevent overfitting. Default is 0.2.
        """
        super(ShipClassifier4ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional layer
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional layer
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth convolutional layer
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=(256*32*32), out_features=512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=2)  # Output 2 classes (ship or no ship)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a PyTorch model.
    
    Args:
        model (`nn.Module`): PyTorch model.
    
    Returns:
        `int`: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model_from_hf(
    repo_id: str = "AdamMuhtar/ShipClassifierConvNet",
    filename: Literal[
        "shipclassifier2convnet.pt", "shipclassifier4convnet.pt"
    ] = "shipclassifier4convnet.pt",
    weights_only: bool = True
) -> nn.Module:
    """
    Loads the ShipClassifierConvNet PyTorch models from Hugging Face Model Hub
    onto the detected device.

    Args:
        repo_id (`str`): Hugging Face repository ID containing the model. Defaults
            to "AdamMuhtar/ShipClassifierConvNet".
        filename (`"shipclassifier2convnet.pt"` or `"shipclassifier4convnet.pt"`):
            Name of the model file to load. Defaults to "shipclassifier4convnet.pt".
        weights_only (`bool`): If True, loads only the model weights. Defaults to
            True.

    Returns:
        torch.nn.Module: The PyTorch model loaded onto the appropriate device.

    Raises:
        RuntimeError: If the model cannot be loaded due to device or file issues.
    """
    # Determine the best available device for PyTorch
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_built()
        else "cpu"
    )
    logging.info(f"Using device: {device}")

    # Download the model file from the Hugging Face Model Hub
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        raise RuntimeError(f"Failed to download model from {repo_id}. Error: {e}")

    # Load the model onto the selected device
    try:
        if filename == "shipclassifier2convnet.pt":
            model = ShipClassifier2ConvNet().to(device)
        else:
            model = ShipClassifier4ConvNet().to(device)
        model.load_state_dict(
            torch.load(
                f=model_path, map_location=device, weights_only=weights_only
            )
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load the model from {model_path}. Error: {e}")

    return model


def preprocess_image(
    image_path: Union[str, Path],
    target_size: tuple = (512, 512),
    device: Literal["cpu", "cuda", "mps"] = "mps"
) -> torch.Tensor:
    """
    Preprocess an image for inference with a PyTorch model.
    
    Args:
        image_path (`str` or `Path`): Path to the image file.
        target_size (`tuple`): Target size for resizing the image. Default is
            (512, 512).
        device (`"cpu"`, `"cuda"` or `"mps"` only): Device to run the inference on.
            Default is "mps".
    
    Returns:
        `torch.Tensor`: Preprocessed image tensor.
    
    Raises:
        FileNotFoundError: If the image file does not exist.
        RuntimeError: If there is an error during preprocessing.
    """
    try:
        # Load the image
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError as e:
        logging.error(f"Image file not found: {image_path}")
        raise e
    except Exception as e:
        logging.error(f"Error loading image: {e}")
        raise RuntimeError(f"Error loading image: {e}")

    try:
        # Resize the image
        transform = transforms.Compose(
            [
                transforms.Resize(size=target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        )
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        logging.error(f"Error during image preprocessing: {e}")
        raise RuntimeError(f"Error during image preprocessing: {e}")

    return image_tensor


def run_inference(
    model: nn.Module,
    image_path: Union[str, Path],
    class_names: List,
    device: Literal["cpu", "cuda", "mps"] = "mps"
) -> Tuple[str, float]:
    """
    Run inference on an image using a PyTorch model.
    
    Args:
        model (`nn.Module`): PyTorch model for inference.
        image_path (`str` or `Path`): Path to the image file.
        class_names (`List`): List of class names.
        device (`"cpu"`, `"cuda"` or `"mps"` only): Device to run the inference
            on. Default is "mps".
    
    Returns:
        `Tuple[str, float]`: Predicted class and confidence score.
    
    Raises:
        RuntimeError: If there is an error during image preprocessing or model
            inference.
    """
    try:
        # Preprocess the image
        image_tensor = preprocess_image(image_path, device=device)
    except Exception as e:
        logging.error(f"Error during image preprocessing: {e}")
        raise RuntimeError(f"Error during image preprocessing: {e}")

    try:
        with torch.no_grad():
            # Set the model to evaluation mode
            model.eval()
            
            # Perform forward pass to get model outputs
            outputs = model(image_tensor)
            
            # Get predicted class by finding the index with the maximum score
            _, predicted = torch.max(outputs, 1)
            
            # Calculate confidence scores using softmax and convert to percentage
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
            
            # Map predicted class index to the class name
            predicted_class = class_names[predicted.item()]
    except Exception as e:
        logging.error(f"Error during model inference: {e}")
        raise RuntimeError(f"Error during model inference: {e}")
    
    return predicted_class, confidence


def visualise_feature_maps(
    model: nn.Module,
    image_path: Union[str, Path],
    layer_name: int = 0,
    device: Literal["cpu", "cuda", "mps"] = "mps",
    fig_size: Tuple[int, int] = (14, 8)
) -> None:
    """
    Visualise feature maps produced by a specified convolutional layer.

    Args:
        model (`nn.Module`): The model to visualize.
        image_path (`str` or `Path`): Path to the input image.
        layer_name (`int`): Name of the layer to visualise. 0 is the first
            convolutional layer, 1 is the second convolutional layer, and so on.
            Default is 0.
        device (`str`): Torch device to use. Default is "mps".
        fig_size (`Tuple[int, int]`): Plot figure size. Default is (14, 8).
    """
    # Ensure the model is in evaluation mode
    model.eval()
    model.to(device)
    
    # Hook to capture feature maps
    feature_maps = None
    def hook_fn(
        module: nn.Module, input: tuple[torch.Tensor], output: torch.Tensor
    ) -> None:
        """
        Hook function to capture the output (feature maps) of a layer during a
        forward pass. Used as a forward hook to intercept the output of a specific
        layer in a neural network. The output (feature maps) is detached from the
        computation graph and stored in a nonlocal variable for later analysis.

        Args:
            module (`nn.Module`): The layer module from which the hook captures
                the output.
            input (`tuple[torch.Tensor]`): The input tensor(s) to the layer. This
                includes any tensors passed to the layer during the forward pass.
            output (`torch.Tensor`): The output tensor of the layer. This is the
                feature map produced by the layer during the forward pass.

        Returns:
            `None`: It updates the nonlocal `feature_maps` variable to store the
                layer's output.
        
        Reference:
            https://www.digitalocean.com/community/tutorials/pytorch-hooks-gradient-clipping-debugging
        """
        # Use nonlocal to update external feature_maps variable
        nonlocal feature_maps

        # Detach output from computation graph to avoid modifying the model's gradients
        feature_maps = output.detach()

    # Register hook for the specified layer
    for name, layer in model.named_modules():
        if str(layer_name) in name:
            layer.register_forward_hook(hook_fn)
            break
    else:
        raise ValueError(f"Layer '{layer_name}' not found in model.")
    
    # Forward pass to capture feature maps
    with torch.no_grad():
        _ = model(preprocess_image(image_path, device=device))
    
    # Move feature maps to CPU for visualisation
    feature_maps = feature_maps.cpu().squeeze(0)
    
    # Plot feature maps
    num_feature_maps = feature_maps.size(0)
    cols = 8
    rows = (num_feature_maps + cols - 1) // cols  # Compute rows dynamically
    
    plt.figure(figsize=fig_size)
    for i in range(num_feature_maps):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(feature_maps[i].numpy(), cmap="viridis")
        plt.axis("off")
        plt.title(f"Filter {i + 1}")
    plt.tight_layout()
    plt.show()
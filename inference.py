#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: inference.py
Author: Adam Muhtar <adam.muhtar23@imperial.ac.uk>
Description: Run ship classification inference on a image via the command line.
"""

# Standard library imports
import argparse
from pathlib import Path
from typing import Literal

# Local application imports
from ship_detection import load_model_from_hf, run_inference, structlog_logger

# Configure logging
logger = structlog_logger()

def main(
    image_path: str ,
    model_filename: Literal["shipclassifier2convnet.pt", "shipclassifier4convnet.pt"],
    device: Literal["cpu", "cuda", "mps"]
) -> None:
    """Run ship classification inference on a test image.
    
    Args:
        image_path (`str`): Path to the test image.
        model_filename (`"shipclassifier2convnet.pt"` or `"shipclassifier4convnet.pt"`):
            ShipClassifierConvNet model to run inference.
        device (`"cpu"`, `"cuda"`, or `"mps"`): Device to run inference on.
    """
    # Set the path to the test image
    image_path = Path(image_path)

    # Load the model from Hugging Face Hub
    model = load_model_from_hf(filename=model_filename)

    # Run inference on the test image
    predicted_class, confidence = run_inference(
        model=model,
        image_path=image_path,
        class_names=["No Ships", "Ships"],
        device=device,
    )
    print(f"Predicted Class: {predicted_class.capitalize()}")
    print(f"Confidence: Ship: {confidence[1].item():.2f}% | No Ship: {confidence[0].item():.2f}%")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Run ship classification inference on a image via the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--image-path", 
        type=str,
        default="data/sfbay_1.png",
        help="Path to the image."
    )
    parser.add_argument(
        "--model", 
        type=Literal["shipclassifier2convnet.pt", "shipclassifier4convnet.pt"],
        choices=["shipclassifier2convnet.pt", "shipclassifier4convnet.pt"],
        default="shipclassifier4convnet.pt", 
        help="ShipClassifierConvNet model to run inference."
    )
    parser.add_argument(
        "--device", 
        type=Literal["cpu", "cuda", "mps"],
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device to run inference on. Choose 'mps' for Apple Silicon processors." 
    )

    # Parse arguments
    args = parser.parse_args()

    # Run the main function with the parsed arguments
    main(
        image_path=args.image_path,
        model_filename=args.model,
        device=args.device
    )
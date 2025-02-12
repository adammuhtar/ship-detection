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

# Local application imports
from ship_detection import load_model_from_hf, run_inference, structlog_logger

# Configure logging
logger = structlog_logger()

def main(image_path: str , model_filename: str) -> None:
    """Run ship classification inference on a test image.
    
    Args:
        image_path (`str`): Path to the test image.
        model_filename (`str`): ShipClassifierConvNet model to run inference.
    """
    # Set the path to the test image
    image_path = Path(image_path)

    # Load the model from Hugging Face Hub
    model = load_model_from_hf(filename=model_filename)

    # Run inference on the test image
    predicted_class, confidence = run_inference(
        model=model,
        image_path=image_path,
        class_names=["No Ships", "Ships"]
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
        "--model-filename", 
        type=str, 
        default="shipclassifier4convnet.pt", 
        help="ShipClassifierConvNet model to run inference."
    )

    # Parse arguments
    args = parser.parse_args()

    # Run the main function with the parsed arguments
    main(args.image_path, args.model_filename)
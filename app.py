#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: app.py
Author: Adam Muhtar <adam.muhtar23@imperial.ac.uk>
Description: FastAPI application to serve the ship detection models.
"""

import os
import tempfile
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ship_detection import (
    __version__,
    load_model_from_hf,
    run_inference,
    structlog_logger,
)

# Configure logging
logger = structlog_logger()

app = FastAPI(
    title="Ship Classification API",
    description="API to run ship classification inference on an uploaded image",
    version=__version__,
)

# Load the model once on startup.
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "shipclassifier4convnet.pt")
model = load_model_from_hf(filename=MODEL_FILENAME)


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File(...)]) -> JSONResponse:
    """Run ship classification inference on an uploaded image. Expects an image
    file sent as a multipart/form-data POST request under the key 'file' and
    returns the predicted class and confidence scores.

    Args:
        file (`UploadFile`): The uploaded image file.

    Returns:
        `JSONResponse`: The predicted class and confidence scores.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    # Save the uploaded file to a temporary file.
    suffix = Path(file.filename).suffix or ".png"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Run inference on the saved image.
        predicted_class, confidence = run_inference(
            model=model,
            image_path=Path(tmp_path),
            class_names=["No Ships", "Ships"],
            device="cpu",
        )
        response = {
            "predicted_class": predicted_class.capitalize(),
            "confidence": {
                "No Ship": float(confidence[0].item()),
                "Ship": float(confidence[1].item()),
            },
        }
    except Exception as e:
        logger.error("Inference failed", error=str(e))
        raise HTTPException(status_code=500, detail="Inference failed.")
    finally:
        # Clean up the temporary file.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return JSONResponse(content=response)

@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint to verify that the service is running."""
    return JSONResponse(content={"status": "ok"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

[![uv](https://img.shields.io/badge/uv-%2350005b?&logo=uv&labelColor=%235A5A5A)](https://docs.astral.sh/uv/getting-started/installation/)
[![Python 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?&logo=Python&logoColor=white%5BPython)](https://www.python.org/downloads/release/python-3129)
[![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-red?&logo=PyTorch&logoColor=white%5BPyTorch)](https://pytorch.org/get-started/locally/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-ShipClassifierConvNet-%23FFCC4D)](https://huggingface.co/AdamMuhtar/ShipClassifierConvNet)
<a href="https://colab.research.google.com/github/adammuhtar/ship-detection/blob/main/notebooks/full_notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Convolutional Neural Network (CNN) to Classify Presence of Ships within Satellite Images
PyTorch code to train and run CNN models to detect presence of ships within satellite imagery.

## Table of Contents
1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Deployment](#deployment)

## Overview
The repo contains code to train and run CNN models to detect the presence of ships within satellite imagery, as part of the final project for Imperial College London's MSc MLDS Unstructured Data Analysis (MATH70103) module. The main files of the repo for the purposes of the module are:
* [full_report.pdf](https://github.com/adammuhtar/ship-detection/blob/main/full_report.pdf): The main project report
* [full_notebook.ipynb](https://github.com/adammuhtar/ship-detection/blob/main/notebooks/full_notebook.ipynb): The Jupyter notebook to reproduce the model training and inference.

We use the [MASATI-v2](https://www.iuii.ua.es/datasets/masati/) (MAritime SATellite Imagery) dataset from [Gallego et al. (2018)](https://www.mdpi.com/2072-4292/10/4/511) to train our models, which can be obtained for free for non-profit research or educational purposes at [https://www.iuii.ua.es/datasets/masati/](https://www.iuii.ua.es/datasets/masati/). We recommend that the unzipped [MASATI-v2](https://www.iuii.ua.es/datasets/masati/) dataset be stored inside the `data` directory for consistency of code (see directory structure).

## Directory Structure
```plaintext
ship-detection/
├── data/
│   ├── MASATI-v2-100mb/ [not provided]
│   └── sfbay_1.png
├── notebooks/
│   ├── full_notebook.ipynb
│   └── test-inference.ipynb
├── src/
│   └── ship_classifier/
│       ├── __init__.py
│       ├── loader.py
│       ├── logger.py
│       ├── model.py
│       └── train.py
├── .dockerignore
├── .gitignore
├── app.py
├── docker-compose.yml
├── Dockerfile
├── final_report.pdf
├── inference.py
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

## Deployment

### Running with Docker
1. We build the image and spin up the container using `docker compose`:
```bash
docker compose up
```
2. Validate that the FastAPI app is running:
```bash
curl http://localhost:5001/health
```
which returns
```bash
{"status":"ok"}
```

1. Make an inference by sending a local image to the endpoint:
```bash
curl -X POST -F "file=@/path/to/image.png" http://localhost:5001/predict
```
which returns an output such as this:
```bash
{"predicted_class":"Ships","confidence":{"No Ship":19.707901000976562,"Ship":80.29209899902344}}
```

### Running without Docker
This code was developed using [Python 3.12.9](https://www.python.org/downloads/release/python-3129/), and compatible with Python versions 3.11 and 3.10. We utilise [uv](https://docs.astral.sh/uv/) to manage Python packages and dependencies for ease-of-use and consistent deployment.

#### Installation via [`uv`](https://docs.astral.sh/uv/)
To install the required packages, clone the repo and install the required dependencies as follows:
```bash
git clone https://github.com/adammuhtar/ship-detection.git
cd ship-detection
uv venv --python 3.12
uv sync --frozen
uv build
uv pip install .
```

Once installed, the command line inference could be run via:
```bash
uv run python inference.py --image-path /path/to/image.png
```

#### Installation via standard Python build
The package could also :
```bash
git clone https://github.com/adammuhtar/ship-detection.git
cd ship-detection
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip build
python -m build
python -m install .
```

Once installed, the command line inference could be run via:
```bash
python inference.py --image-path /path/to/image.png
```
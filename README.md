[![Python 3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?&logo=Python&logoColor=white%5BPython)](https://www.python.org/downloads/release/python-3128)
[![Poetry 1.8.5](https://img.shields.io/badge/Poetry-1.8.5-blue?&logo=Poetry&logoColor=white%5BPython)](https://python-poetry.org/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-red?&logo=PyTorch&logoColor=white%5BPyTorch)](https://pytorch.org/get-started/locally/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-ShipClassifierConvNet-%23FFCC4D)](https://huggingface.co/AdamMuhtar/ShipClassifierConvNet)

# Convolutional Neural Network (CNN) to Classify Presence of Ships within Satellite Images
The repo contains code to train and run ~134 million parameter CNN models to detect the presence of ships within satellite imagery.

## Table of Contents
1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Installation](#installation)

## Directory Structure
```plaintext
ship-detection/
├── .gitignore
├── README.md
├── requirements.txt
├── poetry.lock
├── pyproject.toml
├── images/
│   └── sfbay_1.png
├── notebooks/
│   ├── pretrain-cnn.ipynb
│   └── test-inference.ipynb
└── ship_detection/
    ├── __init__.py
    ├── loader.py
    ├── logger.py
    ├── model.py
    └── train.py
```

## Installation

This code was developed using [Python 3.12.8](https://www.python.org/downloads/release/python-3128/), and compatible with Python versions 3.11 and 3.10. We utilise [Poetry](https://python-poetry.org) to manage Python packages and dependencies for ease-of-use and consistent deployment.

To install the required packages, clone the repo and install the required dependencies as follows:
```bash
git clone https://github.com/adammuhtar/ship-detection.git
cd ship-classifier-cnn
python -m pip install poetry
poetry install
```

We use the [MASATI-v2](https://www.iuii.ua.es/datasets/masati/) (MAritime SATellite Imagery) dataset from [Gallego et al. (2018)](https://www.mdpi.com/2072-4292/10/4/511) to train our models, which can be obtained for free for non-profit research or educational purposes at [https://www.iuii.ua.es/datasets/masati/](https://www.iuii.ua.es/datasets/masati/). We recommend that the unzipped [MASATI-v2](https://www.iuii.ua.es/datasets/masati/) dataset be stored inside a new `data` directory (not included here) for consistency of code.
# AstroSharpener

AI-powered tool for enhancing astronomical galaxy images using deep learning.

## Disclaimer

This is a **learning project** and test implementation made for fun. I'm actively training a new model that will be significantly more accurate. The current version taught me a lot about astronomical image processing and deep learning, and the next iteration will address the current limitations.

**⚠️ Important**: This tool is NOT intended for scientific research or analysis. Do not use enhanced images for scientific measurements or publications. Always use original, unprocessed data for any scientific work.

## About

This is a GUI application that uses a custom neural network to sharpen galaxy images. The model was trained on starless Hubble telescope images to learn enhancement patterns for astronomical photography.

> **Current Status**: This is my first attempt at training a model for this task. While it successfully sharpens galaxy features, it has some limitations. I'm currently training a much more accurate second model based on what I learned from this version.

## Model Performance

| Metric | Value |
|--------|-------|
| Training Epochs | 32 |
| Final MSE | 1.51e-04 |
| Dataset | Thousands of starless Hubble images |

**Known Issue**: May introduce minor noise artifacts due to training data characteristics.

## Features

- **Image Support**: Load FITS and TIFF astronomical images
- **Enhancement Control**: Adjustable strength (0.1× to 2.0×)
- **Comparison View**: Before/after with draggable divider
- **Large Image Processing**: Tiled processing for memory efficiency
- **Background Preservation**: Automatic sky region detection
- **Navigation**: Zoom and pan with synchronized views

## Installation

**Requirements**: Windows, Python 3.8 or higher

### 1. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv310

# Activate virtual environment
venv310\Scripts\activate

### 2. Install Dependencies
```bash
pip install tensorflow>=2.8.0 PyQt5>=5.15.0 numpy opencv-python astropy tifffile matplotlib tqdm

## Usage

```bash
python main.py

1. Load an astronomical image (FITS/TIFF)
2. Adjust enhancement strength and settings
3. Process and compare results in the GUI

### Navigation Controls

- **Mouse Wheel**: Zoom in/out
- **Click + Drag**: Pan image
- **Ctrl + Click**: Reset zoom
- **Right Click**: Zoom options

## Technical Implementation

The application processes images using:

- **Tiled Processing**: Overlapping tiles with Hann window blending
- **Background Detection**: Automatic pure sky region identification
- **Memory Management**: GPU memory growth configuration
- **Real-time GUI**: Synchronized zoom/pan across multiple views

## File Structure
AstroSharpener/
├── main.py              # GUI application entry point
├── processor.py         # Image processing and model handling
├── worker.py            # Background processing thread
├── utils.py             # Image I/O utilities
├── enhanced_loss.py     # Custom loss functions
├── models/              # Trained model files
└── README.md


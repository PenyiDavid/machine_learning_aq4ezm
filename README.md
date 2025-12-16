# Fruits-360 Classification Project

A Convolutional Neural Network (CNN) project for classifying fruits using the Fruits-360 dataset from HuggingFace.

## Overview

This project implements a deep learning model to classify different types of fruits using TensorFlow and Keras. The model uses a CNN architecture with multiple convolutional layers, batch normalization, and dropout for regularization.

## Dataset

- **Source**: [PedroSampaio/fruits-360](https://huggingface.co/datasets/PedroSampaio/fruits-360) from HuggingFace
- **Image Size**: 100x100 pixels
- **Train/Validation Split**: 80/20 (seed: 42)
- **Batch Size**: 32

## Model Architecture

The CNN model consists of:

- **Input Layer**: 100x100x3 RGB images
- **Convolutional Blocks**:
  - Conv2D (32 filters, 3x3) + LeakyReLU(0.4) + BatchNormalization + MaxPooling (3x3)
  - Conv2D (16 filters, 2x2) + LeakyReLU(0.25) + MaxPooling (3x3)
  - Conv2D (8 filters, 2x2) + LeakyReLU(0.25) + MaxPooling (3x3)
  - Conv2D (8 filters, 2x2) + LeakyReLU(0.1) + MaxPooling (2x2)
- **Fully Connected Layers**:
  - Flatten
  - Dense (32 units) + LeakyReLU(0.25)
  - Dropout (0.25)
  - Dense (16 units) + LeakyReLU(0.25)
  - Dense (16 units) + LeakyReLU(0.25)
  - Output: Dense (NUM_CLASSES) + Softmax

## Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Max Epochs**: 100
- **Early Stopping**: 
  - Monitor: Validation Loss
  - Patience: 10 epochs
  - Restore Best Weights: Enabled

## Project Structure

```
project_fruits/
├── main.ipynb          # Main notebook with model training and evaluation
├── models/
│   └── model.keras     # Saved trained model
└── README.md           # Project documentation
```

## Requirements

```
tensorflow
numpy
pandas
datasets (HuggingFace)
matplotlib
```

## Usage

1. Open `main.ipynb` in Jupyter Notebook or VS Code
2. Run all cells sequentially to:
   - Load and preprocess the dataset
   - Build the CNN model
   - Train the model with early stopping
   - Visualize training/validation loss and accuracy
   - Evaluate on test set
   - Save the trained model

## Results

The model is evaluated on the test set, with results showing:
- Test Loss
- Test Accuracy (%)

Training and validation metrics are visualized through loss and accuracy plots.

## Model Saving

The trained model is saved in Keras format to:
```
models/model.keras
```

This allows for easy loading and inference later using:
```python
from tensorflow.keras.models import load_model
model = load_model('models/model.keras')
```

## Key Features

- **Data Augmentation Ready**: Uses TensorFlow data pipeline with prefetching for efficient training
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Batch Normalization**: Improves training stability and convergence
- **LeakyReLU Activation**: Helps prevent dying ReLU problem
- **Dropout Regularization**: Reduces overfitting

## Notes

- Images are normalized to [0, 1] range
- The dataset is streamed efficiently using HuggingFace's datasets library
- Model uses progressive feature extraction with decreasing spatial dimensions

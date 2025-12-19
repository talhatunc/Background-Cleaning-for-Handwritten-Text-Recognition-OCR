# Background Cleaning & OCR for Handwritten Text

This project provides a complete pipeline for **Handwritten Text Recognition (OCR)** using deep learning. It integrates traditional computer vision techniques for background cleaning with a convolutional neural network (CNN) trained on the **EMNIST Balanced** dataset.

## Features

-   **Background Cleaning:** Uses adaptive thresholding and morphological operations to remove noise and isolate text from receipts or paper backgrounds.
-   **Character Segmentation:** Automatically detects and segments individual characters using contour analysis.
-   **Deep Learning Model:** A custom CNN model trained on EMNIST to recognize 47 classes (Digits 0-9 and Letters A-Z).
-   **Preprocessing Pipeline:** Smart padding and resizing to prepare real-world snippets for the neural network.

## Directory Structure

-   `notebooks/`: Contains the Jupyter notebooks for training and execution.
    -   `Background-Cleaning-for-Handwritten-Text-Recognition-OCR.ipynb`: The main pipeline for processing images.
    -   `Train-Model.ipynb`: Notebook to train the EMNIST model from scratch.
-   `models/`: Stores the trained Keras model (`emnist_model.h5`).
-   `handwritten-receipts/`: Input directory for test images (e.g., medical receipts).
-   `all-segmented-outputs/`: Output directory where segmented character images are saved.

## Getting Started

1.  **Train the Model (Optional):**
    If you don't have the model, run the training logic in `Background-Cleaning-for-Handwritten-Text-Recognition-OCR.ipynb` or `Train-Model.ipynb`.

2.  **Run OCR:**
    Place your images in `handwritten-receipts/` and run the main specific notebook. The system will:
    -   Clean the image background.
    -   Detect characters.
    -   Predict the text content.
    -   Visualize the results with bounding boxes and labels.

## Requirements

-   Python 3.x
-   TensorFlow / Keras
-   OpenCV (`opencv-python`)
-   Numpy
-   Matplotlib
-   EMNIST (`emnist` package)

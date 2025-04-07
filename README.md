# Image Captioning Project

This repository contains an image captioning system that generates descriptive captions for images using deep learning techniques.

## Project Overview

The project utilizes a neural network model to analyze images and produce relevant textual descriptions. This involves processing images through a convolutional neural network (CNN) to extract features, which are then used by a recurrent neural network (RNN) to generate captions.

## Repository Structure

- `.github/workflows/`: Contains GitHub Actions workflows for continuous integration and deployment.
- `Image caption-Copy1.ipynb`: Jupyter Notebook demonstrating the image captioning model and its implementation.
- `Mini project.pdf`: Documentation detailing the project's objectives, methodologies, and findings.
- `app2.py`: Python script for running the image captioning application.
- `requirements.txt`: Lists the Python dependencies required to run the project.
- Various pickle files (`.pkl`) and text files (`.txt`) for storing model data, tokenizers, and evaluation metrics.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. Install the necessary packages using:

```bash
pip install -r requirements.txt
```
## Running the Application
To generate captions for your images:

Place your images in the appropriate directory.
```bash
Run the app2.py script:
python app2.py
```

Follow the on-screen instructions to input your image and receive the generated caption.

## Model Architecture
The image captioning model combines a CNN for feature extraction from images and an RNN (specifically, an LSTM) for generating text sequences. The architecture is visualized in the model.png file.

## Evaluation
The model's performance is evaluated using BLEU scores, which are provided in the bleu_scores.txt file.


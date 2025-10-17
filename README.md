# Digit Recognition with Neural Networks

## Table of Contents
- [Project Overview](#project-overview)
- [Models Implemented](#models-implemented)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

***
## Project Overview
This repository contains a series of neural network implementations for handwritten digit recognition, based on the MNIST dataset. The project explores the progression from building a neural network from first principles to implementing and tuning advanced, industry-standard architectures like Convolutional Neural Networks (CNNs) using PyTorch.

The project is structured in four main parts:
1.  **Neural Network from Scratch** using only NumPy.
2.  **Fully-Connected Network** in PyTorch with hyperparameter tuning.
3.  **Convolutional Neural Network (CNN)** in PyTorch.
4.  **Advanced Multi-Digit Recognition** using two-headed MLP and CNN models.

***
## Models Implemented

### 1. Neural Network from Scratch (NumPy)
A simple, fully-connected neural network built using only the NumPy library. This model was created to demonstrate a fundamental understanding of the core mechanics of neural networks, including:
- Forward Propagation
- Backpropagation
- Gradient Descent

### 2. PyTorch Fully-Connected Network (MLP)
A Multi-Layer Perceptron (MLP) built with PyTorch. This module focuses on the practical application of a deep learning framework to establish a baseline and systematically improve performance through:
- **Hyperparameter Tuning**: Experiments were conducted on learning rate, batch size, momentum, and activation functions.
- **Architectural Modification**: The impact of increasing the hidden layer size (from 10 to 128 units) was evaluated.

### 3. PyTorch Convolutional Neural Network (CNN)
A standard CNN architecture for image classification. This model leverages specialized layers to achieve high performance on the MNIST task:
- **Convolutional Layers (`nn.Conv2d`)**: To detect spatial features like edges and patterns.
- **Pooling Layers (`nn.MaxPool2d`)**: To downsample feature maps and create spatial invariance.
- **Dropout**: As a regularization technique to prevent overfitting.

### 4. Multi-Output ("Two-Headed") Architecture
For the advanced multi-digit recognition task, both MLP and CNN models were adapted to have two separate output layers ("heads"). This allows the network to produce two independent predictions (one for each digit) from a single input image.

***
## Dataset

### Source
- **MNIST Dataset**: A large database of 42x28 grayscale images of handwritten single digits (0-9).
- **Multi-Digit MNIST**: A more complex, modified version of MNIST where each image is 42x28 and contains two digits.

### Data Split
For the PyTorch models, the training data was further split into a **training set** (90%) and a **validation set** (10%) to monitor performance and prevent overfitting during hyperparameter tuning.

***
## Installation

### Step 1: Clone the Repository
Clone this repository to your local machine:
```bash
git clone [https://github.com/marlon-sb/deep-learning-mnist.git](https://github.com/marlon-sb/deep-learning-mnist.git)
cd deep-learning-mnist

### Step 2: Create and Activate a Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

### Step 3: Install Dependencies
Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### Usage
The project is organized into modules within the src/ directory.

#### Part 1: NumPy Neural Network
To validate the from-scratch implementation, run the test script from the project root:

```bash
python run_tests.py
```

#### Part 2 & 3: PyTorch Single-Digit Models
To run the fully-connected (MLP) or convolutional (CNN) models for single-digit recognition, execute the respective scripts:

```bash
# Run the fully-connected model
python src/part2_pytorch_fully_connected/nnet_fc.py

# Run the convolutional model
python src/part3_pytorch_convolutional/nnet_cnn.py
```

You can modify hyperparameters directly within these files to experiment.

#### Part 4: PyTorch Multi-Digit Models
To run the advanced two-headed models, execute the scripts from the part4 module:

```bash
# Run the two-headed MLP
python src/part4_advanced_twodigit/mlp.py

# Run the two-headed CNN
python src/part4_advanced_twodigit/conv.py
```

### Results
The project successfully demonstrates the increasing performance of more sophisticated models and the impact of hyperparameter tuning.

#### Single-Digit Classification (PyTorch)
The hyperparameter tuning process revealed key insights:

Baseline MLP (10 hidden units): 92.05% Test Accuracy.  
Tuned MLP (10 hidden units, batch size 64): 92.99% Test Accuracy.  
Baseline MLP (128 hidden units): 97.69% Test Accuracy.  
Tuned MLP (128 hidden units, LeakyReLU): 97.70% Test Accuracy.  
Final CNN Model: 99.17% Test Accuracy, demonstrating the clear superiority of convolutional architectures for image tasks.

Sample output for the final CNN:

```plaintext
-------------
Epoch 10:

Train loss: 0.021677 | Train accuracy: 0.993924
Val loss:   0.038478 | Val accuracy:   0.990976
Loss on test set: 0.027581 Accuracy on test set: 0.991687
```

#### Multi-Digit Classification (PyTorch)
For the final and most complex task, a two-headed CNN was implemented to recognize two digits within a single image. This architecture demonstrated strong performance, achieving high accuracy on the test set for both digits.

Sample output for the two-headed CNN:

```plaintext
Test loss1: 0.072322  accuracy1: 0.980847  
loss2: 0.089698  accuracy2: 0.975806
```

# Task1_MNIST

This repository contains code for training and using various classifiers on the MNIST dataset, as well as some additional helper functions and interfaces.

## Class Descriptions

### `MnistClassifier`
This class takes the name of the classification algorithm (such as `rf`, `nn`, or `cnn`) as an input parameter and provides predictions using the same structure across different algorithms.

### `MnistClassifierRF`
This class implements the Random Forest classifier for MNIST. It uses the `RandomForestClassifier` from `sklearn.ensemble` to classify MNIST digits.

### `MnistClassifierCNN`
This class implements a Convolutional Neural Network (CNN) for MNIST classification. The image pixel values are normalized to the range [0, 1] using the `transforms.ToTensor()` method.

### `MnistClassifierNN`
This class implements a fully connected neural network (MLP) for MNIST classification, using PyTorch. The image pixel values are normalized to the range [0, 1] using the `transforms.ToTensor()` method.

### `MnistClassifierInterface`
This class provides an interface for using the different classifiers in a consistent manner.

## Installation

To install the dependencies, use the following command:

```bash
pip install -r requirements.txt

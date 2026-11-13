# FlowerVision | Deep Learning-Based Flower Classification

<p align="center">
  <strong>An image classification system powered by Convolutional Neural Networks and MobileNetV2 Transfer Learning.</strong>
</p>

<p align="center">
  <a href="https://colab.research.google.com/drive/1spicyBe2NHsH6yPTKB0HBGSp2F6Ou-vZ">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Google Colab">
  </a>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Computer%20Vision-Image%20Classification-blue" alt="Computer Vision">
</p>

---

## Overview

**FlowerVision** is a deep learning-based image classification project that identifies five different categories of flowers from digital images.

The project implements and compares two approaches to image classification:

1. **Custom Convolutional Neural Network (CNN):** A deep learning model built and trained from scratch to learn visual patterns directly from flower images.
2. **MobileNetV2 Transfer Learning:** A pretrained convolutional neural network that uses ImageNet-learned visual representations, followed by task-specific training and fine-tuning for flower classification.

The final model achieves **92.00% test accuracy** using MobileNetV2 transfer learning.

Beyond classification, the project incorporates data augmentation, model checkpointing, learning-rate scheduling, confusion matrix analysis, classification metrics, Grad-CAM visualization, and prediction on user-uploaded images.

**Objective:** Build an image classification pipeline that can recognize flower categories, compare deep learning architectures, and provide interpretable predictions.

## Problem Statement

Identifying flower species manually from images can be time-consuming and requires visual expertise.

The objective of FlowerVision is to develop an automated image classification system that takes a flower image as input and predicts its corresponding category.

The system performs five-class classification:

- Dandelion
- Daisy
- Tulips
- Sunflowers
- Roses

The project explores whether transfer learning using a pretrained architecture can improve classification performance compared with a CNN trained entirely from scratch.

---

## Dataset

**Dataset:** TensorFlow Flowers (`tf_flowers`)

**Source:** [TensorFlow Datasets — TF Flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers)

The dataset is loaded directly using TensorFlow Datasets.

| Dataset property | Value |
|---|---:|
| Total images | 3,670 |
| Number of classes | 5 |
| Image channels | 3 (RGB) |
| Input image dimensions | 160 × 160 |
| Training images | 2,569 |
| Validation images | 551 |
| Test images | 550 |

### Flower Categories

| Class | Description |
|---|---|
| Dandelion | Yellow flowering plant |
| Daisy | White-petaled flower with a yellow center |
| Tulips | Cup-shaped flowering plants |
| Sunflowers | Large flowers with prominent central disks |
| Roses | Flowers belonging to the genus Rosa |

### Dataset Distribution

The dataset is divided into three subsets:

| Subset | Proportion | Images |
|---|---:|---:|
| Training | 70% | 2,569 |
| Validation | 15% | 551 |
| Testing | 15% | 550 |
| **Total** | **100%** | **3,670** |

The training set is used to learn model parameters, the validation set monitors training performance, and the test set evaluates final model performance.

---

## Methodology

The project follows a complete deep learning workflow.

### 1. Environment Setup and Reproducibility

The notebook uses TensorFlow and TensorFlow Datasets alongside NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.

A fixed random seed of `42` is configured for TensorFlow, NumPy, and Python's random module.

This improves the reproducibility of the experiment.

### 2. Data Acquisition

The dataset is loaded using `tfds.load()`.

The notebook creates separate training, validation, and testing datasets directly from the original dataset.

The class names and dataset metadata are retrieved programmatically.

Sample images from the training set are displayed to inspect the visual characteristics of each category.

### 3. Image Preprocessing

All images are resized to a consistent input resolution.

| Parameter | Value |
|---|---:|
| Image height | 160 pixels |
| Image width | 160 pixels |
| Channels | 3 |
| Batch size | 32 |
| Data type after preprocessing | float32 |

The preprocessing pipeline includes:

- Resizing images using `tf.image.resize()`.
- Converting pixel values to floating-point representation.
- Batching images into groups of 32.
- Shuffling the training dataset.
- Using `prefetch(tf.data.AUTOTUNE)` to improve input pipeline efficiency.

### 4. CNN Architecture From Scratch

The first model is a custom Convolutional Neural Network implemented using TensorFlow Keras.

The architecture contains three convolutional blocks followed by fully connected classification layers.

**Architecture:**

| Layer | Configuration |
|---|---|
| Input | 160 × 160 × 3 |
| Rescaling | Pixel normalization to 0–1 |
| Conv2D | 32 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 64 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 128 filters, 3 × 3, ReLU |
| MaxPooling2D | 2 × 2 |
| Flatten | Feature vector |
| Dense | 128 neurons, ReLU |
| Dropout | 0.4 |
| Output | 5 neurons, Softmax |

**Total parameters:** 5,402,437.

The convolutional layers learn increasingly complex visual features, while pooling reduces spatial dimensions.

The final Softmax layer produces class probabilities for the five flower categories.

### 5. CNN Training

The custom CNN is compiled using:

- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Sparse Categorical Crossentropy
- Evaluation metric: Accuracy
- Training epochs: 10

The training and validation accuracy and loss are plotted to examine learning behavior.

**CNN test performance:**

| Metric | Result |
|---|---:|
| Test accuracy | 70.55% |
| Test loss | 1.2904 |

This model provides a baseline for comparison with transfer learning.

### 6. Transfer Learning Using MobileNetV2

The second approach uses **MobileNetV2**, a pretrained convolutional neural network with weights learned from ImageNet.

The architecture is loaded using:

`tf.keras.applications.MobileNetV2`

Configuration:

- Input dimensions: 160 × 160 × 3
- Pretrained weights: ImageNet
- Include top classification layer: False
- Total backbone layers: 154

The original classification head is removed so that the pretrained network can serve as a feature extractor.

Initially, all MobileNetV2 backbone layers are frozen.

This preserves the pretrained visual representations while allowing a new classification head to learn the five flower categories.

### 7. Data Augmentation

Data augmentation is applied during training using a Keras Sequential model.

The transformations include:

| Augmentation | Purpose |
|---|---|
| RandomFlip | Horizontal image variation |
| RandomRotation | Rotational variation |
| RandomZoom | Scale variation |
| RandomContrast | Contrast variation |

These transformations expose the model to different visual variations and can improve generalization.

### 8. Transfer Learning Architecture

The final classification model combines data augmentation, MobileNetV2, feature pooling, regularization, and a custom classification layer.

**Architecture:**

1. Input layer: 160 × 160 × 3.
2. Data augmentation.
3. MobileNetV2 preprocessing.
4. Pretrained MobileNetV2 feature extractor.
5. Global Average Pooling 2D.
6. Dropout with rate 0.3.
7. Dense output layer with five neurons and Softmax activation.

**Model parameters:**

| Parameter category | Count |
|---|---:|
| Total parameters | 2,264,389 |
| Trainable parameters initially | 6,405 |
| Non-trainable parameters initially | 2,257,984 |

Global Average Pooling reduces the spatial feature maps into a compact feature vector before classification.

The Softmax output generates a probability distribution across the five flower categories.

### 9. Transfer Learning Training

The feature extraction model is compiled using:

- Optimizer: Adam
- Learning rate: 0.001
- Loss: Sparse Categorical Crossentropy
- Metric: Accuracy
- Maximum epochs: 12

Three callbacks are implemented:

**EarlyStopping**

Monitors validation loss and stops training if it does not improve for four epochs. The best weights are restored.

**ModelCheckpoint**

Saves the model with the best validation loss as `best_flowervision_model.keras`.

**ReduceLROnPlateau**

Reduces the learning rate when validation loss stops improving, using a reduction factor of 0.2 and a minimum learning rate of 0.0000001.

### 10. Fine-Tuning MobileNetV2

After initial feature extraction, the model undergoes fine-tuning.

The MobileNetV2 backbone is made trainable, but most earlier layers remain frozen.

The notebook:

- Unfreezes the backbone.
- Initially freezes all layers except the last 30.
- Keeps Batch Normalization layers frozen.
- Recompiles the model using a lower learning rate.

Fine-tuning configuration:

| Parameter | Value |
|---|---|
| Learning rate | 0.00001 |
| Maximum epochs | 8 |
| Trainable backbone layers after freezing rules | 19 |
| Loss | Sparse Categorical Crossentropy |
| Optimizer | Adam |

The lower learning rate allows pretrained features to adapt gradually to the flower dataset without drastically disrupting the learned representations.

### 11. Model Comparison

Both models are evaluated on the same held-out test dataset.

| Model | Test Loss | Test Accuracy |
|---|---:|---:|
| CNN From Scratch | 1.2904 | 70.55% |
| MobileNetV2 Transfer Learning | 0.2340 | **92.00%** |

**Key observation:** MobileNetV2 transfer learning achieves substantially better test accuracy than the custom CNN trained from scratch.

The result demonstrates the practical benefit of using pretrained visual representations when working with a relatively small image dataset.

---

## Final Model Performance

The fine-tuned MobileNetV2 model achieves the following results on the 550-image test set.

| Metric | Score |
|---|---:|
| Test Accuracy | 92.00% |
| Weighted Precision | 92.02% |
| Weighted Recall | 92.00% |
| Weighted F1-score | 91.99% |
| Test Loss | 0.2340 |

**Prediction summary:**

- Total test images: 550
- Correct predictions: 506
- Incorrect predictions: 44

### Class-wise Performance

| Flower category | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Dandelion | 93.86% | 95.54% | 94.69% | 112 |
| Daisy | 87.63% | 89.47% | 88.54% | 95 |
| Tulips | 91.54% | 88.81% | 90.15% | 134 |
| Sunflowers | 92.45% | 96.08% | 94.23% | 102 |
| Roses | 94.17% | 90.65% | 92.38% | 107 |

The model achieves its strongest class-wise F1-score on Dandelions, while Daisy has the lowest class-wise F1-score among the five categories.

---

## Model Explainability — Grad-CAM

FlowerVision includes a Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.

Grad-CAM helps visualize which spatial regions of an image contribute most strongly to a model's prediction.

The implementation:

1. Extracts feature maps from MobileNetV2's final convolutional activation layer.
2. Uses `tf.GradientTape` to calculate gradients of the predicted class score.
3. Computes channel-wise importance weights by averaging gradients.
4. Combines feature maps using the calculated weights.
5. Applies ReLU to retain positive contributions.
6. Normalizes the resulting heatmap.

The generated heatmap provides a visual explanation of the regions that influence the model's classification.

This adds an interpretability component to the image classification pipeline.

---

## Custom Image Prediction

The notebook supports predictions on user-uploaded images through Google Colab.

The inference workflow:

1. Upload an image using the Colab file uploader.
2. Convert the image to RGB.
3. Resize it to 160 × 160.
4. Convert it into a NumPy array.
5. Generate class probabilities using the trained model.
6. Select the class with the highest probability.
7. Display the predicted flower and confidence score.

The notebook demonstrates this functionality using a rose image.

**Example prediction:**

| Output | Result |
|---|---|
| Predicted category | Roses |
| Confidence | 99.97% |

The confidence value represents the model's predicted Softmax probability, not a guarantee that the prediction is correct.

### Class probability output

The prediction function also displays the probability assigned to every flower category, allowing users to inspect the model's output distribution.

---

## Model Saving and Loading

The final trained model is saved in Keras format:

`FlowerVision_Final.keras`

The notebook demonstrates:

- Saving the trained model using `model.save()`.
- Downloading the model from Google Colab.
- Loading the saved model using `tf.keras.models.load_model()`.

This allows the trained model to be reused without repeating the entire training process.

---

## Technology Stack

| Technology | Application |
|---|---|
| Python | Core programming language |
| TensorFlow | Deep learning framework |
| Keras | Neural network architecture and training |
| TensorFlow Datasets | Dataset acquisition |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Classification metrics and evaluation |
| Google Colab | Development and execution environment |

---

## Repository Structure

```text
FlowerVision/
│
├── notebooks/
│   └── FlowerVision.ipynb
│
├── models/
│   └── FlowerVision_Final.keras
│
├── README.md
│
└── LICENSE
```

The model file is generated by the notebook. The `models/` directory should be added when the trained model is uploaded to the repository.

---

## Installation and Execution

### Option 1 — Google Colab (Recommended)

1. Open the notebook using the Google Colab badge at the top.
2. Select **Runtime → Run all**.
3. Allow the notebook to install and import its dependencies.
4. The TensorFlow Flowers dataset is downloaded automatically.
5. Execute the complete workflow to train, evaluate, and save the models.

A GPU-enabled Colab runtime is recommended for faster training.

### Option 2 — Local Environment

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/FlowerVision.git
cd FlowerVision
```

Install the dependencies:

```bash
pip install tensorflow tensorflow-datasets numpy pandas matplotlib seaborn scikit-learn
```

Launch Jupyter:

```bash
pip install notebook
jupyter notebook notebooks/FlowerVision.ipynb
```

Run the notebook sequentially.

A compatible Python environment and sufficient computational resources are required. GPU acceleration is recommended for training.

---

## Key Concepts Demonstrated

- Computer vision and image classification
- Convolutional Neural Networks
- Convolution and feature extraction
- Max pooling
- Flattening and dense layers
- ReLU and Softmax activation
- Dropout regularization
- Image preprocessing
- Data augmentation
- Transfer learning
- ImageNet pretrained models
- MobileNetV2 architecture
- Feature extraction
- Fine-tuning
- Batch Normalization
- Adam optimization
- Sparse Categorical Crossentropy
- Early stopping
- Model checkpointing
- Learning-rate scheduling
- Confusion matrix
- Precision, recall, and F1-score
- Grad-CAM and model explainability
- Model serialization and inference

---

## Limitations and Future Improvements

### Current Limitations

- The model is trained on five flower categories from the TensorFlow Flowers dataset.
- Performance on flower species outside these categories has not been evaluated.
- The dataset is relatively small compared with large-scale image classification datasets.
- High Softmax confidence does not guarantee prediction correctness.
- Grad-CAM provides an approximate explanation of model attention, not a complete account of its decision-making process.
- The notebook demonstrates inference in Google Colab rather than a deployed production application.

### Future Improvements

- Expand the dataset to include more flower species.
- Evaluate performance on independent external flower datasets.
- Add top-3 predictions and confidence-based uncertainty handling.
- Implement model quantization for lightweight inference.
- Deploy the model through a FastAPI service.
- Develop an interactive web application for image uploads and predictions.
- Add systematic robustness and error analysis.
- Explore additional architectures such as EfficientNet and ResNet.
- Compare frozen feature extraction against alternative fine-tuning strategies.

---

## References

1. [TensorFlow Flowers Dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers)
2. [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
3. [TensorFlow Documentation](https://www.tensorflow.org/guide)
4. [Keras Documentation](https://keras.io/)
5. [Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization](https://arxiv.org/abs/1610.02391)

---

<p align="center">
  <strong>FlowerVision</strong><br>
  Deep Learning | Computer Vision | Transfer Learning
</p>

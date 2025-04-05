# Machine Learning Classification Pipeline

This project is a two-part machine learning pipeline demonstrating the use of both tabular and image data for classification tasks.

## 📦 Project Structure

### Part 1: E-Commerce Revenue Prediction (Tabular Data)
- **Dataset**: Preprocessed `shopping.csv` file representing user behavior on an e-commerce site.
- **Goals**:
  - Predict whether a user will generate revenue (`Revenue` column).
  - Compare performance of custom linear regression (with and without regularization) and a deep neural network.
- **Techniques**:
  - Data normalization: Min-Max, Z-score, and Mean Normalization.
  - Manual implementation of gradient descent for logistic regression.
  - TensorFlow/Keras neural network with 4 dense layers (last using sigmoid for binary classification).
  - Evaluation metrics: Accuracy, Precision, Recall, F1 Score.
  - Tested on a separate unseen dataset (`unseen.csv`).

### Part 2: Traffic Sign Classification (Image Data)
- **Dataset**: German Traffic Sign Recognition Benchmark (GTSRB).
- **Goals**:
  - Classify traffic signs into 43 categories based on image input.
- **Techniques**:
  - Manual image loading and resizing using OpenCV.
  - Image flattening and normalization.
  - Deep Neural Network using TensorFlow/Keras:
    - Input layer: Flatten
    - Two dense layers with L2 regularization and ReLU activation
    - Output layer with softmax activation
  - Model trained on 60% of data and evaluated on the remaining 40%.

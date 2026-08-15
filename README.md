# 🌿 PlantGuard AI — Plant Disease Detection

An AI-powered plant disease detection system built using deep learning and computer vision.

The application uses a **MobileNetV2 transfer learning model** trained on the **PlantVillage dataset** to classify plant leaf images into **38 plant/disease classes**.

The trained model is integrated into a **Streamlit web application** where users can upload a leaf image and receive the predicted disease, confidence score, and top-3 predictions.

---

## 🚀 Live Demo

**Live Application:**  
Add your Streamlit deployment URL here after deployment.

---

## 📌 Project Overview

Plant diseases can significantly affect crop productivity and agricultural output.

This project explores how computer vision and deep learning can be used to automatically classify plant diseases from leaf images.

The system:

1. Accepts a plant leaf image.
2. Preprocesses the image.
3. Uses a MobileNetV2-based deep learning model.
4. Predicts the most likely plant/disease class.
5. Displays the prediction confidence.
6. Shows the top 3 predictions.
7. Warns users when the model has low confidence.

---

## ✨ Features

- 🌱 Plant disease image classification
- 🧠 MobileNetV2 transfer learning
- 🖼️ Image upload through Streamlit
- 📊 Top-3 predictions
- 📈 Confidence scores
- ⚠️ Low-confidence detection
- 🌿 Healthy leaf classification
- 🎨 Professional Streamlit interface
- 📱 Responsive application layout
- 💾 Saved Keras model
- 📓 Complete Jupyter Notebook workflow

---

## 🧠 Model Architecture

The project uses **MobileNetV2**, a convolutional neural network pretrained on ImageNet.

The pretrained feature extractor is followed by a custom classification head:

```text
Input Image
    │
    ▼
224 × 224 × 3
    │
    ▼
MobileNetV2
Pretrained on ImageNet
    │
    ▼
Global Average Pooling
    │
    ▼
Batch Normalization
    │
    ▼
Dropout
    │
    ▼
Dense Layer
256 neurons
    │
    ▼
Dropout
    │
    ▼
Softmax
38 Classes
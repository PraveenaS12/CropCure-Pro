# CropCure Pro 🌿

**AI-Powered Plant Disease Detector**  
An intuitive, multilingual web app that helps farmers and researchers identify diseases in crop leaves using deep learning.

---

## Table of Contents

- [About](#about)
- [Features](#features)
- [Demo](#demo)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## About

**CropCure Pro** is an AI-powered tool designed to quickly detect plant leaf diseases from uploaded images. Targeted for farmers and agricultural professionals, it utilizes a trained convolutional neural network (CNN) to recognize common diseases and provides actionable cure tips. The user interface is simple, multilingual, and mobile-friendly, built with [Streamlit](https://streamlit.io/).

---

## Features

- 📷 **Image Upload & Prediction:** Upload a photo of a plant leaf and receive instant disease diagnosis with top-3 confidences.
- 💡 **Cure Suggestions:** Get actionable tips and disease info based on the prediction.
- 🌐 **Multilingual:** Disease names, info, and tips in English, Tamil, and Hindi.
- 📊 **Visual Feedback:** Bar charts, emojis, and diagnosis cards for an engaging user experience.
- 📄 **PDF Report:** Project report
- 🎬 **Video Demo:** [Watch the demo here](#demo).

---

## Demo

[![Watch Video Demo](CropCure Pro 🌿 - Personal - Microsoft​ Edge 2025-07-16 19-26-44.mp4)

---

## Dataset

- **Source:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Attribution:** Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016) "Using Deep Learning for Image-Based Plant Disease Detection".

*Note: The dataset is NOT included in this repository due to size and license. Please download it from the [Kaggle link above](https://www.kaggle.com/datasets/emmarex/plantdisease) if you wish to retrain the model.*

---

## Installation

1. **Clone this repository:**
    ```
    git clone https://github.com/PraveenaS12/CropCure-Pro.git
    cd CropCure-Pro
    ```
2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```
    - Or install main packages: `streamlit tensorflow keras pillow pandas fpdf`
3. **Download pre-trained model:**
   - If `CropCure_model.h5` is NOT present (or is too large for GitHub), [Download from Google Drive](YOUR_MODEL_LINK) or [other link].
   - Place `CropCure_model.h5` in the project root.
4. **(Optional) Prepare dataset for retraining:**
   - Download [PlantVillage Dataset from Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
   - Unzip into `./dataset/` for training scripts.

---

## Usage

1. **Run the web app locally:**
    ```
    streamlit run app.py
    ```
2. **For model re-training:**
    - Open `CropCure_model.ipynb` in Jupyter/Colab.
    - Ensure `dataset/` is present and formatted as per [PlantVillage class folders].
    - After training, save your model as `CropCure_model.h5`.

---

## Project Structure

CropCure-Pro/
│
├── app.py # Streamlit web app source
├── CropCure_model.ipynb # Model training Jupyter notebook
├── CropCure_model.h5 # Trained Keras model (see above)
├── README.md # This file
├── CropCure_Pro_Project_Report.pdf # Final project report
├── assets/ # (Optional: images/icons)
└── (dataset/) # Not included - download separately

---

## License

[MIT License](LICENSE)  
*This project is for academic/demo purposes only. Please do not use the diagnosis app for medical or critical agriculture decisions without expert validation.*

---

## Acknowledgements

- [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)  
  Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection.
- [Streamlit](https://streamlit.io/), TensorFlow, and the open source ML community.
- Elevate Labs – Internship Project

> **Project by Praveena S, [Elevate Labs Internship].**

---


> Developed during my internship at Elevate Labs. 🌟

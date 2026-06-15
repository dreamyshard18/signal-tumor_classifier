# 🧠 Brain Tumor MRI Classifier

A deep learning-based web application that classifies brain MRI images into tumor categories using an EfficientNetB0 model.

---

## 🚀 Overview

This project uses transfer learning with EfficientNetB0 to classify brain MRI scans into four categories:

* Glioma Tumor
* Meningioma Tumor
* Pituitary Tumor
* No Tumor

The model is deployed through a simple Flask web application with an interactive UI that allows users to upload MRI images and view predictions along with confidence scores.

---

## 📊 Features

* Upload MRI image through web interface
* Predict tumor type using trained deep learning model
* Display confidence score
* Show top-2 predictions
* Visual probability bars for all classes
* Clean and modern UI

---

## 🧠 Model Details

* Base Model: EfficientNetB0 (pretrained on ImageNet)
* Input Size: 224 × 224
* Transfer Learning + Fine-tuning
* Loss Function: Sparse Categorical Crossentropy
* Optimizer: Adam

---

## 📁 Dataset

This model was trained on the Brain Tumor MRI dataset by Masoud Nickparvar:

👉 https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The dataset contains labeled MRI images across four classes.

---

## 🖥️ Demo

### Before Prediction

![Before Upload](assets/1.jpg)

### After Prediction

![After Prediction](assets/2.jpg)

---

## ⚙️ Project Structure

```
Brain-Tumor-Classifier/
│
├── app.py
├── brain_tumor_efficientnetb0.keras
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── notebook.ipynb
```

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* Flask
* HTML, CSS, JavaScript

---

## ▶️ How to Run

1. Clone the repository

```
git clone https://github.com/your-username/brain-tumor-classifier.git
cd brain-tumor-classifier
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the Flask app

```
python app.py
```

4. Open in browser

```
http://127.0.0.1:5000
```

---

## ⚠️ Notes

* Model performs well on dataset images but may vary on real-world MRI scans due to domain differences.
* This project is for educational and experimental purposes only.
---

## 📌 Future Improvements

* Improve generalization on real-world MRI data
* Add more robust preprocessing
* Deploy using Hugging Face / cloud platform


---


# Braintumor_with_gradcam_upgradation


# 🧠 Brain Tumor Classification Using Deep Learning & Explainable AI

A Flask-based web application for **brain tumor classification from MRI images** using **deep learning models (VGG16 and EfficientNet)** with **Grad-CAM based explainability**.  
The system provides prediction results, confidence scores, and visual heatmaps highlighting tumor regions.

---

## 📌 Project Overview

Brain tumor detection from MRI images is a critical task in medical diagnosis. Manual analysis is time-consuming and depends heavily on expert knowledge.  
This project automates the detection process using **Convolutional Neural Networks (CNNs)** and enhances trust by integrating **Explainable AI (Grad-CAM)**.

The application is deployed as a **Flask web app**, allowing users to upload MRI images and receive instant predictions.

---

## 🎯 Objectives

- Automate brain tumor classification from MRI images  
- Compare performance of **VGG16** and **EfficientNet-B0**  
- Provide confidence scores for predictions  
- Generate Grad-CAM heatmaps for explainability  
- Reject non-MRI images using validation logic  
- Build a user-friendly web interface using Flask  

---

## 🛠️ Technologies Used

- **Programming Language:** Python  
- **Framework:** Flask  
- **Deep Learning:** TensorFlow / Keras  
- **Models:** VGG16, EfficientNet-B0  
- **Explainable AI:** Grad-CAM  
- **Frontend:** HTML, CSS  
- **Image Processing:** OpenCV, PIL  
- **Environment:** Virtual Environment (venv)

---

## 🧠 Models Used

### 🔹 VGG16
- Deep CNN with 16 layers  
- Strong feature extraction  
- High accuracy but computationally heavy  

### 🔹 EfficientNet-B0
- Optimized and lightweight CNN  
- Faster inference  
- Suitable for real-time web applications  

Both models use **transfer learning** with pretrained ImageNet weights.

---

## 🔍 System Architecture

1. User uploads MRI image via web interface  
2. Flask backend receives the request  
3. Image validation (MRI / non-MRI check)  
4. Image preprocessing (resize, normalize)  
5. Prediction using deep learning model  
6. Grad-CAM heatmap generation  
7. Display result, confidence score, and heatmap  

---

## 🔥 Grad-CAM Based Explainability

Grad-CAM highlights regions of the MRI image that influenced the model’s prediction.  
It improves transparency by showing **where the model is focusing**, making the system suitable for medical decision support.

---

## 🚀 How to Run the Project Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/SUBIRSARKAR100/brain-tumor-classification.git
cd brain-tumor-classification


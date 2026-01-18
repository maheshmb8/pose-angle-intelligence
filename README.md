# Pose Angle Intelligence

An end-to-end computer vision pipeline to classify human image orientation using
pose estimation and multimodal embeddings.

This project combines **MoveNet pose keypoints**, **geometric feature engineering**,
**CLIP image embeddings**, **PCA dimensionality reduction**, and a
**LightGBM multiclass classifier** to predict the orientation / angle of a person
in an image.

---

## 🔍 Problem Statement

Given a human image, determine the **pose / orientation angle** of the subject
(front, side, back, etc.) using a robust, interpretable, and scalable ML pipeline.

The system is designed to:
- Work on **real-world product images**
- Handle **partial visibility and occlusions**
- Generalize to **completely unseen images**

---

## 🧠 Solution Overview

### 1. Pose Estimation (MoveNet)
- Extract 17 human keypoints per image
- Includes x, y coordinates and confidence scores

### 2. Pose Feature Engineering
- 40+ engineered features:
  - Limb lengths & ratios
  - Torso geometry & angles
  - Visibility & confidence metrics
  - Symmetry and alignment indicators

### 3. CLIP Image Embeddings
- Generate 512-dimensional image embeddings using CLIP (ViT-B/32)
- Capture global visual context beyond pose

### 4. Dimensionality Reduction
- Apply PCA to CLIP embeddings (32 components)
- Reduce noise and improve model stability

### 5. Model Training
- Multiclass LightGBM classifier
- Trained on labeled pose angle data
- Evaluated using accuracy, confusion matrix, and error analysis

---

## 📁 Repository Structure


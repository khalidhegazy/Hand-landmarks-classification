# Hand-landmarks-classification

## Overview
This project implements a hand landmarks classification system using machine learning techniques. It leverages the Mediapipe library for hand landmark detection and various machine learning models for gesture classification.

## Dataset
The dataset used consists of hand landmark coordinates extracted using Mediapipe from the HaGRID Dataset. It includes 21 keypoints for each hand, with x,y and z coordinates, and labels representing different hand gestures.

## Features
- **Data Preprocessing:**
  - Label encoding
  - Data normalization using MinMaxScaler and StandardScaler
  - Handling missing values and duplicates
  
- **Machine Learning Models:**
  - Random Forest Classifier (Grid Search Optimization)
  - Support Vector Classifier (Grid Search Optimization) ✅ Best Model (Accuracy: 97.9%)
  - XGBoost Classifier (Grid Search Optimization)
  - Extra Trees Classifier (Grid Search Optimization)
  - K-Nearest Neighbors (Grid Search Optimization)

- **Visualization:**
  - 2D and 3D visualization of hand landmarks
  - Confusion matrix for model evaluation

- **Deployment:**
  - Live video processing using OpenCV and Mediapipe
  - Real-time gesture recognition and annotation

### Required Libraries
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `xgboost`
- `imblearn`
- `joblib`
- `opencv-python`
- `mediapipe`
- `seaborn`
- `tqdm`

## MLflow Experiment Tracking

All training experiments are tracked using MLflow, including:

Parameters and hyperparameters
Accuracy, precision, recall, F1-score
Confusion matrices
Model artifacts (saved binaries)
Preprocessing pipeline objects

## Model Performance
| Model | Accuracy |
|--------|----------|
| Random Forest | 85.7% |
| SVC (Best) | 97.9% |
| XGBoost | 91.6% |
| Extra Trees | 81.7% |
| KNN | 64.9% |

![Mlflow](https://github.com/user-attachments/assets/8bc8d881-b197-48ef-86ef-19b39f2b949e)

## Results
The SVC model achieved the highest accuracy (97.9%) and was used for deployment. The system can classify 18 different hand gestures accurately in real-time.

## Output Video
![Video_output](https://github.com/user-attachments/assets/af3e5cc8-03a4-42f9-834e-946ef5cb2fd7)

## Author
Eng. Khalid Ahmed Mohamed


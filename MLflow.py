import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import mlflow
import mlflow.sklearn
import os
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("hand_landmarks_data.csv")
data = data.drop(columns=[col for col in data.columns if 'z' in col])

# Encode labels
label_encoder = LabelEncoder()
data["label"] = label_encoder.fit_transform(data["label"])
features=data.iloc[:,:-1]
labels=data.iloc[:,-1]

# Split data to train & test
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.1, random_state=42, stratify=labels)

# Apply StandardScaler
MM_scaler = MinMaxScaler()
features_train = MM_scaler.fit_transform(features_train)
features_test = MM_scaler.transform(features_test)

# Save the scaler for later use in the API
joblib.dump(MM_scaler, "MMscale.pkl")

mlflow.set_experiment("HandGesture_Best_SVC_Training")
best_C = 200
best_gamma = 'scale'
best_kernel = 'poly'

with mlflow.start_run(run_name="Best_SVC_Model_Run"):
    # Log best parameters for this specific run
    mlflow.log_param("SVC_C", best_C)
    mlflow.log_param("SVC_gamma", best_gamma)
    mlflow.log_param("SVC_kernel", best_kernel)

    # Train the SVC model with the best parameters
    svm_model = SVC(C=200,gamma='scale', kernel='poly') 
    svm_model.fit(features_train, labels_train)

    # Evaluate the model
    train_predictions = svm_model.predict(features_train)
    test_predictions = svm_model.predict(features_test)

    train_accuracy = accuracy_score(labels_train, train_predictions)
    test_accuracy = accuracy_score(labels_test, test_predictions)
    test_precision = precision_score(labels_test, test_predictions, average='weighted', zero_division=0)
    test_recall = recall_score(labels_test, test_predictions, average='weighted', zero_division=0)
    test_f1 = f1_score(labels_test, test_predictions, average='weighted', zero_division=0)

    # Log metrics
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1_score", test_f1)

    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Generate and log Confusion Matrix as an artifact
    cm = confusion_matrix(labels_test, test_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_encoder.inverse_transform(sorted(labels.unique())),
                yticklabels=label_encoder.inverse_transform(sorted(labels.unique())))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix for Best SVC Model")
    plt.tight_layout()
    cm_path = "confusion_matrix_best_svc.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")
    plt.close() # Close plot to prevent display issues in non-interactive environments
    os.remove(cm_path) # Clean up local file

    # Log the trained SVC model as an artifact
    mlflow.sklearn.log_model(svm_model, "best_svm_model_artifact")

    # Log the StandardScaler as an artifact
    scaler_filename = "stscale.pkl"
    # The scaler was already saved with joblib.dump above, so just log it.
    mlflow.log_artifact(scaler_filename, artifact_path="preprocessing_scalers")
    # os.remove(scaler_filename) # Only remove if you're sure you won't need it locally

    print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
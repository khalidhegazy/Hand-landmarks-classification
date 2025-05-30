# %%
pip install mediapipe

# %% [markdown]
# # Import libraries

# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import utils
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import cv2
import numpy as np
import mediapipe as mp
import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# # Reading Data

# %%
data=pd.read_csv("hand_landmarks_data.csv")

# %% [markdown]
# # Visualize Data

# %% [markdown]
# 3D

# %%
sample = data.iloc[5, :-1] 
x_values = sample[::3].values  
y_values = sample[1::3].values  
z_values = sample[2::3].values  

hand_connections = [
    (0, 1), (1, 2), (2, 3), (3, 4), 
    (0, 5), (5, 6), (6, 7), (7, 8),  
    (5, 9), (9, 10), (10, 11), (11, 12), 
    (9, 13), (13, 14), (14, 15), (15, 16), 
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20) 
]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')


ax.scatter(x_values, y_values, z_values, color='red', s=50, label="Landmarks")

for connection in hand_connections:
    ax.plot(
        [x_values[connection[0]], x_values[connection[1]]],
        [y_values[connection[0]], y_values[connection[1]]],
        [z_values[connection[0]], z_values[connection[1]]],
        color='blue'
    )

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

ax.invert_zaxis()

plt.show()


# %%
data.head(5)

# %%
data = data.drop(columns=[col for col in data.columns if 'z' in col])

# %% [markdown]
# 2D 

# %%
sample = data.iloc[1, :-1]  
x_values = sample[::2].values 
y_values = sample[1::2].values  
hand_connections = [
    (0, 1), (1, 2), (2, 3), (3, 4), 
    (0, 5), (5, 6), (6, 7), (7, 8),  
    (5, 9), (9, 10), (10, 11), (11, 12), 
    (9, 13), (13, 14), (14, 15), (15, 16), 
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20) 
]
plt.figure(figsize=(6, 6))
plt.scatter(x_values, y_values, color='red', s=50, label="Landmarks")
for connection in hand_connections:
    plt.plot(
        [x_values[connection[0]], x_values[connection[1]]],
        [y_values[connection[0]], y_values[connection[1]]],
        color='blue'
    )
plt.gca().invert_yaxis()
plt.show()


# %% [markdown]
# ### Encoding labels

# %%
label_encoder = LabelEncoder()
data["label"] = label_encoder.fit_transform(data["label"])

# %%
data["label"].value_counts()

# %%
data.isna().sum()

# %%
data.info()

# %%
data.duplicated().sum()

# %%
data.shape

# %% [markdown]
# # Split The Data

# %%
features=data.iloc[:,:-1]
labels=data.iloc[:,-1]

# %% [markdown]
# Split data to train & test

# %%
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.1, random_state=42, stratify=labels)

# %%
MM_scaler = MinMaxScaler()
features_train = MM_scaler.fit_transform(features_train)
features_test = MM_scaler.transform(features_test)

# %%
joblib.dump(MM_scaler, "MMscale.pkl")

# %%
st_scaler = StandardScaler()
features_train = st_scaler.fit_transform(features_train)
features_test = st_scaler.transform(features_test)

# %%
joblib.dump(st_scaler, "stscale.pkl")

# %% [markdown]
# # RandomForestClassifier with Grid Search

# %%

param_grid = {
    'n_estimators': [50, 100, 200, 300], 
    'max_depth': [10, 20, 30, None],  
    'min_samples_split': [2, 5, 10],   
    'min_samples_leaf': [1, 2, 4]      
}
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(features_train, labels_train)

# %% [markdown]
# RandomForestClassifier best hyperparameters

# %%
print("Best Parameters:", grid_search_rf.best_params_)
print("Best Validation Accuracy:", grid_search_rf.best_score_)

# %% [markdown]
# RandomForestClassifier best model Train

# %%
random_forest_classifier =RandomForestClassifier(n_estimators=300,max_depth=30)
random_forest_classifier.fit(features_train, labels_train)

# %% [markdown]
# Test

# %%
test_predictions = random_forest_classifier.predict(features_test)
test_accuracy = accuracy_score(labels_test, test_predictions)
print(f"Test Accuracy: {test_accuracy:.4f}")

# %% [markdown]
# # SVC with Grid Search

# %%
param_grid = {
    'C': [0.01, 0.1, 1, 10, 50, 100, 200], 
    'gamma': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 'scale', 'auto'],
    'kernel': ['rbf', 'poly', 'sigmoid'],
}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(features_train, labels_train)

# %%
print("Best Parameters:", grid_search.best_params_)
print("Best Validation Accuracy:", grid_search.best_score_)

# %% [markdown]
# SVC best model

# %%
svm_model = SVC(C=200,gamma='scale', kernel='poly')  
svm_model.fit(features_train, labels_train)

# %% [markdown]
# Test

# %%
test_predictions = svm_model.predict(features_test)
test_accuracy = accuracy_score(labels_test, test_predictions)
print(f"Test Accuracy: {test_accuracy:.4f}")

# %% [markdown]
# Confusion Matrix

# %%
cm = confusion_matrix(labels_test, test_predictions)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# %% [markdown]
# Save the SVC best weights

# %%
joblib.dump(svm_model, "SVM_model.pkl")

# %% [markdown]
# Load the model

# %%
svm = joblib.load("svm_model.pkl")

# %% [markdown]
# Test the loaded model

# %%
test_predictions = svm.predict(features_test)
test_accuracy = accuracy_score(labels_test, test_predictions)
print(f"Test Accuracy: {test_accuracy:.4f}")

# %% [markdown]
# # XGBClassifier with Grid Seatch

# %%
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model = XGBClassifier(objective='multi:softmax', num_class=18, eval_metric='mlogloss', use_label_encoder=False)

grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(features_train, labels_train)

# %% [markdown]
# XGB best hyperparameter

# %%
print("Best Parameters:", grid_search.best_params_)
print("Best Validation Accuracy:", grid_search.best_score_)

# %% [markdown]
# XGBClassifier best model
# 

# %%
xgb_model = XGBClassifier(
    colsample_bytree=1.0, 
    gamma=0, 
    learning_rate=0.2, 
    max_depth=5, 
    n_estimators=500, 
    subsample=0.8,
    objective='multi:softmax', 
    num_class=18, 
    eval_metric='mlogloss', 
    use_label_encoder=False
)

xgb_model.fit(features_train, labels_train)

# %% [markdown]
# Test

# %%
test_predictions = xgb_model.predict(features_test)
test_accuracy = accuracy_score(labels_test, test_predictions)
print(f"Test Accuracy: {test_accuracy:.4f}")

# %%
joblib.dump(xgb_model, "xgb_model.pkl")

# %% [markdown]
# # ExtraTreesClassifier with Grid Search

# %%
param_grid = {
    'n_estimators': [50, 100, 200, 300], 
    'max_depth': [10, 20, 30, None],     
    'min_samples_split': [2, 5, 10],      
    'min_samples_leaf': [1, 2, 4]         
}
grid_search_etc = GridSearchCV(ExtraTreesClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_etc.fit(features_train, labels_train)

# %% [markdown]
# ExtraTreesClassifier best hyperparameters

# %%
print("Best Parameters:", grid_search_etc.best_params_)
print("Best Validation Accuracy:", grid_search_etc.best_score_)

# %% [markdown]
# # KNN with Grid Search

# %%
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],  
    'weights': ['uniform', 'distance'], 
    'metric': ['euclidean', 'manhattan', 'minkowski']  
}
grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_knn.fit(features_train, labels_train)

# %% [markdown]
# KNN best hyperparameters

# %%
print("Best Parameters:", grid_search_knn.best_params_)
print("Best Validation Accuracy:", grid_search_knn.best_score_)

# %% [markdown]
# # Deployment

# %%
svm_model = joblib.load("SVM_model.pkl")
scaler = joblib.load("stscale.pkl") 

hagrid_classes = {
    0: "call", 1: "dislike", 2: "fist", 3: "four", 4: "like",
    5: "mute", 6: "ok", 7: "one", 8: "palm", 9: "peace",
    10: "peace inv.", 11: "rock", 12: "stop", 13: "stop inv.",
    14: "three", 15: "three 2", 16: "two up", 17: "two up inv."
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_drawing = mp.solutions.drawing_utils

video_path = "Video.mp4"
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

output_path = "Video_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

progress_bar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

feature_names = [f"x{i//2+1}" if i % 2 == 0 else f"y{i//2+1}" for i in range(42)]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x * frame_width, lm.y * frame_height])  

            keypoints_array = np.array(keypoints).reshape(1, -1)
            if keypoints_array.shape[1] == 42:
                keypoints_scaled = scaler.transform(keypoints_array)  

                keypoints_df = pd.DataFrame(keypoints_scaled, columns=feature_names)

                prediction_idx = svm_model.predict(keypoints_df)[0]
                predicted_class = hagrid_classes.get(int(prediction_idx), "Unknown")

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, predicted_class, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    out.write(frame)
    progress_bar.update(1)
cap.release()
out.release()
progress_bar.close()


# %%




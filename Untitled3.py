#!/usr/bin/env python
# coding: utf-8

# In[34]:


import os
import numpy as np
import librosa
import glob
from tensorflow.keras.models import load_model
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Load model and label encoder ---
MODEL_PATH = 'model3.keras'
LABEL_ENCODER_PATH = 'label_encoder.pkl'

model = load_model(MODEL_PATH)
with open(LABEL_ENCODER_PATH, 'rb') as f:
    le = pickle.load(f)

# --- Feature extraction function ---
def extract_mfcc_sequence(file_path, n_mfcc=40, max_len=200):
    y, sr = librosa.load(file_path, res_type='scipy')
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T

# --- Prediction function ---
def predict_emotion(file_path, model, le, max_len=200):
    mfcc_seq = extract_mfcc_sequence(file_path, max_len=max_len)
    mfcc_seq = np.expand_dims(mfcc_seq, axis=0)
    pred = model.predict(mfcc_seq)
    predicted_class = np.argmax(pred)
    return le.classes_[predicted_class]

# --- Main Testing Section ---
RANDOM_TEST_DIR = '/content/drive/MyDrive/Speech/'  # Change to your folder path

audio_files = sorted(glob.glob(os.path.join(RANDOM_TEST_DIR, '*.wav')))
num_files = len(audio_files)
if num_files == 0:
    print(f"No audio files found in {RANDOM_TEST_DIR}")
    exit()

print(f"Found {num_files} audio files in {RANDOM_TEST_DIR}")

# --- Automatically generate y_true from filenames using RAVDESS convention ---
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
y_true = []
for file_path in audio_files:
    file_name = os.path.basename(file_path)
    parts = file_name.split('-')
    if len(parts) > 2:
        emotion_code = parts[2]
        label = emotion_map.get(emotion_code, 'unknown')
        y_true.append(label)
    else:
        y_true.append('unknown')

# --- Predict and Evaluate ---
results = []
y_pred = []
for file_path in audio_files:
    emotion = predict_emotion(file_path, model, le)
    results.append((os.path.basename(file_path), emotion))
    y_pred.append(emotion)
    print(f"{os.path.basename(file_path)}: {emotion}")

if len(y_true) == num_files:
    y_true_enc = le.transform(y_true)
    y_pred_enc = le.transform(y_pred)
    acc = accuracy_score(y_true_enc, y_pred_enc)
    cm = confusion_matrix(y_true_enc, y_pred_enc)
    print(f"\nAccuracy on random dataset: {acc*100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    labels = list(range(len(le.classes_)))
    print(classification_report(y_true_enc, y_pred_enc, labels=labels, target_names=le.classes_, zero_division=0))
    print(confusion_matrix(y_true_enc, y_pred_enc, labels=labels))

else:
    print("\nTrue labels not provided or length mismatch; skipping accuracy and confusion matrix calculation.")
    print(f"Expected {num_files} labels, got {len(y_true)}.")


# In[ ]:





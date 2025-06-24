
# Speech Emotion Recognition on RAVDESS

## Project Description

This project implements a deep learning pipeline for Speech Emotion Recognition (SER) using the RAVDESS dataset.  
The pipeline is built and trained in Google Colab, leveraging audio augmentation and a CNN-BiLSTM architecture to classify emotions from speech.  
Final model evaluation is performed using accuracy and confusion matrix metrics.

---

## Pre-processing Methodology

**Data Loading:**  
- All `.wav` files are loaded from RAVDESS folders in Google Drive using glob patterns.

**Emotion Extraction:**  
- Emotion labels are parsed from the third field of each filename and mapped via:  
  `'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'`

**Data Augmentation:**  
For each audio file, three augmentations are applied:
- Additive noise
- Time shifting
- Time stretching (using `librosa.effects.time_stretch`)

**Feature Extraction:**  
- MFCCs (Mel Frequency Cepstral Coefficients) are extracted for each audio, zero-padded or truncated to a fixed length (200 frames, 40 coefficients).

**Label Encoding:**  
- Emotions are label-encoded and saved with `pickle` as `label_encoder.pkl`.

---

## Model Pipeline

**Architecture:**
- 1D Convolutional layers (Conv1D + BatchNorm + MaxPooling + Dropout)
- Bidirectional LSTM
- Dense layers with Dropout
- Output: softmax over 8 emotion classes

**Training:**
- Early stopping on validation loss
- Class weighting to handle class imbalance
- Data split: 80% training, 20% test (stratified)

**Evaluation:**
- Metrics: accuracy, classification report, confusion matrix (from sklearn)

**Saving:**
- Model is saved as `model3.keras`
- Label encoder saved as `label_encoder.pkl`

---

## Accuracy Metrics

### Test Accuracy

- **0.9399** (93.99%) on the held-out test set.

### Classification Report

| Class      | Precision | Recall | F1-score | Support |
|------------|-----------|--------|----------|---------|
| angry      | 0.97      | 0.99   | 0.98     | 301     |
| calm       | 0.93      | 0.93   | 0.93     | 301     |
| disgust    | 0.93      | 0.90   | 0.91     | 154     |
| fearful    | 0.96      | 0.94   | 0.95     | 301     |
| happy      | 0.95      | 0.95   | 0.95     | 301     |
| neutral    | 0.94      | 0.89   | 0.91     | 150     |
| sad        | 0.89      | 0.94   | 0.91     | 301     |
| surprised  | 0.93      | 0.93   | 0.93     | 153     |
| **accuracy**  |           |        | 0.94     | 1962    |
| **macro avg** | 0.94      | 0.93   | 0.94     | 1962    |
| **weighted avg** | 0.94   | 0.94   | 0.94     | 1962    |

### Confusion Matrix

|           | angry | calm | disgust | fearful | happy | neutral | sad | surprised |
|-----------|-------|------|---------|---------|-------|---------|-----|-----------|
| **angry**     | 297   | 0    | 2       | 2       | 0     | 0       | 0   | 0         |
| **calm**      | 0     | 281  | 2       | 2       | 1     | 2       | 13  | 0         |
| **disgust**   | 6     | 1    | 138     | 0       | 0     | 1       | 5   | 3         |
| **fearful**   | 2     | 0    | 2       | 284     | 2     | 0       | 8   | 3         |
| **happy**     | 1     | 5    | 2       | 0       | 286   | 1       | 4   | 2         |
| **neutral**   | 0     | 10   | 0       | 0       | 4     | 133     | 1   | 2         |
| **sad**       | 0     | 4    | 2       | 6       | 4     | 3       | 282 | 0         |
| **surprised** | 0     | 0    | 0       | 1       | 4     | 2       | 3   | 143       |

---
**Demo Deployed Link:** [[https://emotionclassificationmodel-ejqtfoxibqkqwekahsg7p5.streamlit.app]](https://emotionclassificationmodel-ejqtfoxibqkqwekahsg7p5.streamlit.app)

## References

- [RAVDESS Zenodo page](https://zenodo.org/records/1188976)
- [RAVDESS on Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- [RAVDESS PLoS ONE paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196391)

---

**Cite RAVDESS as:**  
Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391

---

*This README documents the full pipeline, pre-processing, model, and evaluation for your RAVDESS SER project as implemented in your Colab notebook.*

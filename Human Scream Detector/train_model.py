import os
import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Data augmentation function
def augment_audio(y, sr):
    # Time stretching
    y_stretch = librosa.effects.time_stretch(y, rate=1.1)  # Speed up
    # Pitch shifting
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)  # Higher pitch
    return [y, y_stretch, y_pitch]

# Define paths
positive_dir = 'positive/positive'
negative_dir = 'negative/negative'

# Function to extract MFCC and additional features from audio files
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    augmented_audios = augment_audio(y, sr)
    features_list = []

    for y_aug in augmented_audios:
        mfccs = librosa.feature.mfcc(y=y_aug, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        chroma = librosa.feature.chroma_stft(y=y_aug, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        spectral_contrast = librosa.feature.spectral_contrast(y=y_aug, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
        
        features = np.hstack((mfccs_mean, chroma_mean, spectral_contrast_mean))
        features_list.append(features)

    return np.mean(features_list, axis=0)

# Prepare the dataset
X = []
y = []

# Process positive samples
for file in os.listdir(positive_dir):
    if file.endswith('.wav'):
        file_path = os.path.join(positive_dir, file)
        features = extract_features(file_path)
        X.append(features)
        y.append(1)  # Label for scream (positive)

# Process negative samples
for file in os.listdir(negative_dir):
    if file.endswith('.wav'):
        file_path = os.path.join(negative_dir, file)
        features = extract_features(file_path)
        X.append(features)
        y.append(0)  # Label for non-scream (negative)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVC model
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'scream_detection_model.pkl')

# Calculate and save accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
with open('accuracy.txt', 'w') as f:
    f.write(str(accuracy))

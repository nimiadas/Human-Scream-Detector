from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import librosa
import traceback

app = Flask(__name__)

try:
    model = joblib.load('scream_detection_model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('accuracy.txt', 'r') as f:
        accuracy = f.read()
    print(f"Model Accuracy: {accuracy}")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    traceback.print_exc()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            y, sr = librosa.load(file, sr=None)
            features = extract_features(y, sr)
            features = features.reshape(1, -1)
            features = scaler.transform(features)
            prediction = model.predict(features)

            result = "Scream detected" if prediction[0] == 1 else "No scream detected"
            return jsonify({'result': result})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
    
    return np.hstack((mfccs_mean, chroma_mean, spectral_contrast_mean))

if __name__ == '__main__':
    app.run(debug=True)

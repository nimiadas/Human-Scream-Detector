# Human Scream Detector

The **Human Scream Detector** is a real-time application that detects human screams using machine learning, aimed at enhancing safety and emergency response capabilities. This tool uses audio feature extraction to classify sounds as either "scream" or "non-scream." The model is built using a Support Vector Machine (SVM) classifier and deployed through a web interface using Flask. Users can upload or record audio, and the model will provide an immediate prediction on whether the sound is a scream.

## Project Structure

- **train_model.py**: This script is responsible for training the SVM model. It includes:
  - **Data Collection**: Loads and labels audio samples for screams and non-screams.
  - **Feature Extraction**: Extracts Mel-Frequency Cepstral Coefficients (MFCC) from the audio.
  - **Model Training**: Trains the SVM model on the extracted features and saves the model and scaler for future use.
  
- **main.py**: This file contains the Flask application. It serves a web interface that allows users to:
  - **Upload or Record Audio**: Users can submit audio files to be analyzed.
  - **Predict Scream or Non-Scream**: The app loads the saved model and scaler, extracts features from the audio input, and returns a prediction.
  
- **index.html**: The HTML file for the web interface where users can interact with the scream detector.
  
- **script.js**: Contains JavaScript functions for handling user interactions, including file uploads and displaying prediction results.

This project structure allows for easy interaction with the trained model, providing a user-friendly experience for real-time scream detection.

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Feature columns matching the CSV structure
FEATURE_COLUMNS = [
    'age', 'fever', 'cough', 'fatigue', 'headache', 'breathing_difficulty',
    'chest_pain', 'muscle_stiffness', 'pain', 'seizures', 'fatigue.1',
    'b_difficulty'
]

class ModelTrainer:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.le_diagnosis = LabelEncoder()
        self.le_severity = LabelEncoder()
        self.diagnosis_model = None
        self.severity_model = None

        os.makedirs('models', exist_ok=True)

    def load_and_preprocess_data(self):
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded data with columns: {df.columns.tolist()}")

            df['diagnosis_encoded'] = self.le_diagnosis.fit_transform(df['diagnosis'])
            df['severity_encoded'] = self.le_severity.fit_transform(df['severity'])

            X = df[FEATURE_COLUMNS]
            y_diagnosis = df['diagnosis_encoded']
            y_severity = df['severity_encoded']

            return X, y_diagnosis, y_severity
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def train_models(self):
        try:
            X, y_diagnosis, y_severity = self.load_and_preprocess_data()
            X_train, _, y_train_diagnosis, _ = train_test_split(
                X, y_diagnosis, test_size=0.2, random_state=42
            )
            _, _, y_train_severity, _ = train_test_split(
                X, y_severity, test_size=0.2, random_state=42
            )

            self.diagnosis_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.severity_model = RandomForestClassifier(n_estimators=100, random_state=42)

            self.diagnosis_model.fit(X_train, y_train_diagnosis)
            self.severity_model.fit(X_train, y_train_severity)

            self.save_models()
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise

    def save_models(self):
        try:
            joblib.dump(self.diagnosis_model, 'models/diagnosis_model.pkl')
            joblib.dump(self.severity_model, 'models/severity_model.pkl')
            joblib.dump(self.le_diagnosis, 'models/diagnosis_encoder.pkl')
            joblib.dump(self.le_severity, 'models/severity_encoder.pkl')
            logger.info("Models and encoders saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise

class PredictionService:
    def __init__(self):
        self.diagnosis_model = joblib.load('models/diagnosis_model.pkl')
        self.severity_model = joblib.load('models/severity_model.pkl')
        self.le_diagnosis = joblib.load('models/diagnosis_encoder.pkl')
        self.le_severity = joblib.load('models/severity_encoder.pkl')

    def validate_input(self, data):
        try:
            feature_mapping = {
                'age': 'age',
                'fever': 'fever',
                'cough': 'cough',
                'fatigue': 'fatigue',
                'headache': 'headache',
                'breathing_difficulty': 'breathing_difficulty',
                'chest_pain': 'chest_pain',
                'muscle_stiffness': 'muscle_stiffness',
                'pain': 'pain',
                'seizures': 'seizures',
                'fatigue_cp': 'fatigue.1',
                'breathing_difficulty_cp': 'b_difficulty'
            }

            features = [float(data.get(api_name, 0)) for api_name in feature_mapping.keys()]
            return features
        except ValueError as e:
            raise ValueError(f"Invalid input data: {str(e)}")

    def predict(self, features):
        try:
            diagnosis_pred = self.diagnosis_model.predict([features])[0]
            severity_pred = self.severity_model.predict([features])[0]

            diagnosis = self.le_diagnosis.inverse_transform([diagnosis_pred])[0]
            severity = self.le_severity.inverse_transform([severity_pred])[0]

            return {
                'diagnosis': diagnosis,
                'severity': severity,
                'confidence': {
                    'diagnosis': float(max(self.diagnosis_model.predict_proba([features])[0])),
                    'severity': float(max(self.severity_model.predict_proba([features])[0]))
                }
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
@app.route('/', methods=['GET'])
def home():
    """Render the index.html template."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        prediction_service = PredictionService()
        features = prediction_service.validate_input(data)
        prediction = prediction_service.predict(features)

        return jsonify(prediction)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

def initialize_app():
    try:
        trainer = ModelTrainer('data/combined_symptoms_150.csv')
        trainer.train_models()
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        raise

if __name__ == '__main__':
    initialize_app()
    app.run(host='0.0.0.0', port=5000, debug=False)

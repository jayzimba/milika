from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the CSV file for training
def load_data():
    data = pd.read_csv('combined_symptoms_150.csv')
    return data

# Train the diagnosis and severity models (including CP and Epilepsy)
def train_models(data):
    # General conditions diagnosis model
    X_diagnosis = data[['age', 'fever', 'cough', 'fatigue', 'headache', 'breathing_difficulty', 'chest_pain']]
    y_diagnosis = data['diagnosis']

    X_train_diag, X_test_diag, y_train_diag, y_test_diag = train_test_split(X_diagnosis, y_diagnosis, test_size=0.2, random_state=42)

    model_diagnosis = RandomForestClassifier(random_state=42)
    model_diagnosis.fit(X_train_diag, y_train_diag)

    # Severity model for CP, Epilepsy, and other conditions
    X_severity = data[['muscle_stiffness', 'pain', 'seizures', 'fatigue', 'b_difficulty']]
    y_severity = data['severity']

    X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(X_severity, y_severity, test_size=0.2, random_state=42)

    model_severity = RandomForestClassifier(random_state=42)
    model_severity.fit(X_train_sev, y_train_sev)

    return model_diagnosis, model_severity

# Load data and train models when the application starts
data = load_data()
model_diagnosis, model_severity = train_models(data)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the POST request
    input_data = request.get_json()

    # Extract features for diagnosis prediction
    diagnosis_features = [[
        input_data['age'],
        input_data['fever'],
        input_data['cough'],
        input_data['fatigue'],
        input_data['headache'],
        input_data['breathing_difficulty'],
        input_data['chest_pain']
    ]]

    # Predict diagnosis (could be general condition, CP, or Epilepsy)
    diagnosis_prediction = model_diagnosis.predict(diagnosis_features)[0]

    # Extract features for severity prediction (includes CP and Epilepsy symptoms)
    severity_features = [[
        input_data['muscle_stiffness'],
        input_data['pain'],
        input_data['seizures'],
        input_data['fatigue'],
        input_data['breathing_difficulty']
    ]]

    # Predict severity
    severity_prediction = model_severity.predict(severity_features)[0]

    # Return predictions as a JSON response
    return jsonify({
        'diagnosis': diagnosis_prediction,
        'severity': severity_prediction
    })

if __name__ == '__main__':
    app.run(debug=True)

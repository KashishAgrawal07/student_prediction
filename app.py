from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os

app = Flask(__name__)
CORS(app)

# Global variables for model and scaler
model = None
scaler = None

def create_sample_data():
    """Create sample student data for training"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features: attendance, study_hours, previous_grade, assignments_completed
    attendance = np.random.uniform(40, 100, n_samples)
    study_hours = np.random.uniform(0, 10, n_samples)
    previous_grade = np.random.uniform(30, 100, n_samples)
    assignments_completed = np.random.uniform(0, 100, n_samples)
    
    # Create target variable (Pass/Fail) based on features
    # Pass if weighted score > 60
    weighted_score = (
        0.3 * attendance + 
        0.25 * study_hours * 10 + 
        0.3 * previous_grade + 
        0.15 * assignments_completed
    )
    
    # Add some randomness
    noise = np.random.normal(0, 5, n_samples)
    weighted_score += noise
    
    # 1 = Pass, 0 = Fail
    result = (weighted_score > 60).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'attendance': attendance,
        'study_hours': study_hours,
        'previous_grade': previous_grade,
        'assignments_completed': assignments_completed,
        'result': result
    })
    
    return data

def train_model():
    """Train the ANN model"""
    global model, scaler
    
    print("Creating sample data...")
    data = create_sample_data()
    
    # Prepare features and target
    X = data[['attendance', 'study_hours', 'previous_grade', 'assignments_completed']].values
    y = data['result'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build the ANN model
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(4,)),
        layers.Dropout(0.2),
        layers.Dense(8, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    print("Training model...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Save model and scaler
    model.save('student_model.h5')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return test_accuracy

def load_model():
    """Load the trained model and scaler"""
    global model, scaler
    
    if os.path.exists('student_model.h5') and os.path.exists('scaler.pkl'):
        model = keras.models.load_model('student_model.h5')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("Model loaded successfully")
    else:
        print("No saved model found. Training new model...")
        train_model()

@app.route('/')
def index():
    """Serve the frontend"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on student data"""
    try:
        data = request.json
        
        # Extract features
        features = np.array([[
            float(data['attendance']),
            float(data['study_hours']),
            float(data['previous_grade']),
            float(data['assignments_completed'])
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction_prob = model.predict(features_scaled, verbose=0)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0
        
        # Prepare response
        result = {
            'prediction': 'Pass' if prediction == 1 else 'Fail',
            'probability': float(prediction_prob * 100),
            'confidence': float(abs(prediction_prob - 0.5) * 200)  # Confidence score
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain the model"""
    try:
        accuracy = train_model()
        return jsonify({
            'message': 'Model retrained successfully',
            'accuracy': float(accuracy * 100)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Load or train model on startup
    load_model()
    
    # Run the app
    app.run(debug=True, port=5000)

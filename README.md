# Student Pass/Fail Prediction System

A web application that predicts whether a student will pass or fail based on their performance metrics using an Artificial Neural Network (ANN).

## Features

- **Backend**: Flask API with TensorFlow/Keras ANN model
- **Frontend**: Interactive HTML/CSS/JavaScript interface
- **Real-time Predictions**: Instant pass/fail prediction with probability scores
- **User-friendly**: Slider-based input for easy data entry

## Model Features

The ANN model uses 4 input features:
1. **Attendance (%)**: Student's class attendance percentage
2. **Study Hours per Day**: Average daily study hours (0-10)
3. **Previous Grade (%)**: Previous academic performance
4. **Assignments Completed (%)**: Percentage of assignments submitted

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install flask flask-cors numpy pandas scikit-learn tensorflow
```

### Step 2: Run the Application

```bash
python app.py
```

The application will:
- Automatically create sample training data
- Train the ANN model (if no saved model exists)
- Start the Flask server on `http://localhost:5000`

## Usage

1. Open your web browser and navigate to `http://localhost:5000`
2. Adjust the sliders for:
   - Attendance percentage
   - Study hours per day
   - Previous grade
   - Assignments completed percentage
3. Click "Predict Result"
4. View the prediction (Pass/Fail) with probability and confidence scores

## Model Architecture

```
Input Layer (4 features)
    ↓
Dense Layer (16 neurons, ReLU activation)
    ↓
Dropout (20%)
    ↓
Dense Layer (8 neurons, ReLU activation)
    ↓
Dropout (20%)
    ↓
Output Layer (1 neuron, Sigmoid activation)
```

## API Endpoints

### `GET /`
Serves the frontend HTML interface

### `POST /predict`
Makes a prediction based on student data

**Request Body:**
```json
{
    "attendance": 85.0,
    "study_hours": 6.0,
    "previous_grade": 75.0,
    "assignments_completed": 90.0
}
```

**Response:**
```json
{
    "prediction": "Pass",
    "probability": 85.34,
    "confidence": 70.68
}
```

### `POST /retrain`
Retrains the model with new sample data

**Response:**
```json
{
    "message": "Model retrained successfully",
    "accuracy": 89.5
}
```

## File Structure

```
├── app.py                      # Flask backend application
├── templates/
│   └── index.html             # Frontend interface
├── requirements.txt           # Python dependencies
├── student_model.h5           # Saved trained model (generated)
└── scaler.pkl                 # Saved feature scaler (generated)
```

## Model Training

The model is trained on 1000 synthetic student records with the following logic:
- Pass/Fail is determined by a weighted score of all features
- Training/Test split: 80/20
- 50 epochs with validation split
- Typical accuracy: ~85-90%

## Customization

### Modify Input Features
Edit the feature list in `app.py`:
```python
X = data[['attendance', 'study_hours', 'previous_grade', 'assignments_completed']].values
```

### Adjust Model Architecture
Modify the neural network structure:
```python
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(4,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

### Use Real Data
Replace `create_sample_data()` function to load your actual student dataset:
```python
def load_real_data():
    data = pd.read_csv('student_data.csv')
    return data
```

## Troubleshooting

**Issue**: Model not loading
- **Solution**: Delete `student_model.h5` and `scaler.pkl`, restart the application

**Issue**: Port 5000 already in use
- **Solution**: Change the port in `app.py`: `app.run(debug=True, port=5001)`

**Issue**: TensorFlow installation errors
- **Solution**: Use CPU-only version: `pip install tensorflow-cpu`

## Future Improvements

- Add more input features (test scores, extracurricular activities, etc.)
- Implement user authentication for student tracking
- Add data visualization dashboard
- Export prediction reports as PDF
- Implement model versioning and A/B testing
- Add real-time model retraining with new data

## License

This project is open source and available for educational purposes.

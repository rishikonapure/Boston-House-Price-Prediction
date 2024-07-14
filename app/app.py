from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from form
        input_features = [float(x) for x in request.form.values()]
        # Scale the input features
        input_features_scaled = scaler.transform([input_features])
        # Predict using the model
        prediction = model.predict(input_features_scaled)[0]
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

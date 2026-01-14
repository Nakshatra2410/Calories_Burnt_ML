ask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')

# Load the XGBoost model
try:
    model = joblib.load('model.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route('/', methods=['GET', 'POST'])
def home():
    # If it's a POST request, handle the form submission
    if request.method == 'POST':
        return predict()  # Call your predict function
    # If it's a GET request, just show the form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Check if model is loaded
            if model is None:
                return render_template('index.html', 
                    prediction_text='Error: Model not loaded. Please check server logs.')
            
            # Capture the 7 inputs from the HTML form
            features = [
                float(request.form['gender']),
                float(request.form['age']),
                float(request.form['height']),
                float(request.form['weight']),
                float(request.form['duration']),
                float(request.form['heart_rate']),
                float(request.form['body_temp'])
            ]
            
            # Convert to Numpy array and Reshape
            final_features = np.array(features).reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(final_features)
            output = float(prediction[0])
            
            return render_template('index.html', 
                prediction_text=f'🔥 Estimated Calories Burnt: {output:.2f} kcal')

        except Exception as e:
            return render_template('index.html', 
                prediction_text=f'Error: Please check all inputs are valid numbers. Details: {str(e)}')

@app.route('/health')
def health():
    return {'status': 'healthy', 'model_loaded': model is not None}

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)  # debug=False for production

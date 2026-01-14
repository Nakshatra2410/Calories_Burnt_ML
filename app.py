from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the XGBoost model you saved from Colab
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Capture the 7 inputs from the HTML form
            # Order: Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp
            features = [
                float(request.form['gender']),
                float(request.form['age']),
                float(request.form['height']),
                float(request.form['weight']),
                float(request.form['duration']),
                float(request.form['heart_rate']),
                float(request.form['body_temp'])
            ]
            
            # 2. Convert to Numpy array and Reshape (just like you did in Colab)
            final_features = np.array(features).reshape(1, -1)
            
            # 3. Make prediction
            prediction = model.predict(final_features)
            output = float(prediction[0])

            return render_template('index.html', prediction_text=f'Estimated Calories Burnt: {output} kcal')

        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
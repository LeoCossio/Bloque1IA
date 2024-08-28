# Python libraries
from flask import Flask, request, jsonify, render_templates
import numpy as np
from load import joblib
# File management
import os
from werkzeug.utils import secure_filename

# Load model
dt = joblib.load("dt.joblib")

# Create flask app
server = Flask(__name__)

# Define a route to send JSON data
@server.route('/predictjson', methods = ['POST'])

def predictjson():
    # Process input data
    data = request.json # Get request
    print(data)

    inputData = np.array([
        data['pH'],
        data['sulphates'],
        data['alcohol']
    ])

    # Predict using the input information and the model
    result = dt.predict(inputData.reshape(1,-1))

    # Send response
    return jsonify({'Prediction': str(result[0])})

if __name__ == '__main__':
    server.run(debug=False, host='0.0.0.0', port=8080)

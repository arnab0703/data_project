import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import json
import pickle
import logging
import time

from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the model
regmodel = pickle.load(open('california_housing_model.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    logger.info(f"Received data for prediction: {data}")
    input_array = np.array(list(data.values())).reshape(1, -1)
    logger.info(f"Reshaped input array: {input_array}")

    new_data = scalar.transform(input_array)

    # Measure response time
    start_time = time.time()
    output = regmodel.predict(new_data)
    response_time = time.time() - start_time

    logger.info(f"Prediction output: {output[0]} with response time: {response_time:.4f} seconds")

    return jsonify({'prediction': output[0], 'response_time': response_time})

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    logger.info(f"Received form data for prediction: {data}")
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    logger.info(f"Transformed input for prediction: {final_input}")

    # Measure response time
    start_time = time.time()
    output = regmodel.predict(final_input)[0]
    response_time = time.time() - start_time

    logger.info(f"Prediction output: {output} with response time: {response_time:.4f} seconds")

    return render_template("home.html", prediction_text="The analysis is {}".format(output), response_time=response_time)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import logging

# Load models
rf_model = pickle.load(open("pickle/water_quality_model_rf.pkl", 'rb'))
svm_model = pickle.load(open("pickle/water_quality_model_svm.pkl", 'rb'))

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    try:
        # Extracting form content separately
        ph = float(request.form['ph'])
        solids = float(request.form['solids'])
        hardness = float(request.form['hardness'])
        conductivity = float(request.form['conductivity'])
        turbidity = float(request.form['turbidity'])
        features = [ph, solids, hardness, conductivity, turbidity]
        features = np.array(features).reshape(1, -1)
        
        app.logger.debug(f"Features : {features}")
        
        rfPrediction = rf_model.predict(features)
        svmPrediction = svm_model.predict(features)
        combined_model = (0.5 * rfPrediction[0] + 0.5 * svmPrediction[0])
        
        app.logger.debug(f"Random Forest Prediction: {rfPrediction}")
        app.logger.debug(f"SVM Prediction: {svmPrediction}")
        app.logger.debug(f"Combined Model Prediction: {combined_model}")
        
        binary_combined_preds = np.round(combined_model)
        
        if binary_combined_preds == 0:
            result = {'Prediction': 'Non-Drinkable'}
        else:
            result = {'Prediction': 'Drinkable'}
        
        # Returning the output back to the HTML
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
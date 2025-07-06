from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('grid_model.pkl')
soil_enc = joblib.load('soil_enc.pkl')
crop_enc = joblib.load('crop_enc.pkl')
fer_enc = joblib.load('fer_enc.pkl')

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp = float(request.form['Temparature'])
    humidity = float(request.form['Humidity'])
    moisture = float(request.form['Moisture'])
    soil_type = request.form['Soil_Type']
    crop_type = request.form['Crop_Type']
    potassium = float(request.form['Potassium'])
    phosphorous = float(request.form['Phosphorous'])
    fertilizer = request.form['Fertilizer_Name']

    soil_encoded = soil_enc.transform([soil_type])[0]
    crop_encoded = crop_enc.transform([crop_type])[0]
    fer_encoded = fer_enc.transform([fertilizer])[0]
    
    input_dict = {
        'Temparature': [temp],
        'Humidity': [humidity],
        'Moisture': [moisture],
        'Soil Type': [soil_encoded],
        'Crop Type': [crop_encoded],
        'Potassium': [potassium],
        'Phosphorous': [phosphorous],
        'Fertilizer Name': [fer_encoded]
    }

    input_df = pd.DataFrame(input_dict)
    prediction = model.predict(input_df)[0]

    if prediction < 10:
        advice = "Nitrogen is low. Consider using nitrogen-rich fertilizers like Urea or compost manure."
    elif 10 <= prediction <= 20:
        advice = "Nitrogen level is adequate for most crops. Maintain organic matter and monitor regularly."
    else:
        advice = "Nitrogen is high. Avoid over-fertilization to prevent leaching and plant stress."

    return render_template('form.html',
                           prediction_text=f"Predicted Nitrogen Content: {prediction:.2f}",
                           advice_text=advice)


if __name__ == "__main__":
    app.run(debug=True)

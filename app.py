from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved models
with open("time_model.pkl", "rb") as f:
    time_model = pickle.load(f)

with open("success_model.pkl", "rb") as f:
    success_model = pickle.load(f)

# Feature columns that were used for training the models
# Ensure these columns match exactly the feature set used in your training data
feature_columns = [
    "Weapon Speed (m/s)", "Target Distance (m)",
    "Weapon Type_Missile", "Weapon Type_Shotgun", "Weapon Type_Cannon", 
    "Weapon Type_Drone", "Weapon Type_Interceptor", "Target Type_Aircraft", 
    "Target Type_Bunker", "Target Type_Infantry", "Target Type_Missile"
]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    weapon_speed = float(request.form['Weapon Speed'])
    target_distance = float(request.form['Target Distance'])
    
    # Assuming form data contains these fields from the dropdown
    weapon_type = request.form['Weapon Type']  # 'Cannon', 'Missile', 'Shotgun', etc.
    target_type = request.form['Target Type']  # 'Aircraft', 'Bunker', 'Infantry', etc.

    # Prepare the one-hot encoding for Weapon Type
    weapon_type_missile = 1 if weapon_type == 'Missile' else 0
    weapon_type_shotgun = 1 if weapon_type == 'Shotgun' else 0
    weapon_type_cannon = 1 if weapon_type == 'Cannon' else 0
    weapon_type_drone = 1 if weapon_type == 'Drone' else 0
    weapon_type_interceptor = 1 if weapon_type == 'Interceptor' else 0
    
    # Prepare the one-hot encoding for Target Type
    target_type_aircraft = 1 if target_type == 'Aircraft' else 0
    target_type_bunker = 1 if target_type == 'Bunker' else 0
    target_type_infantry = 1 if target_type == 'Infantry' else 0
    target_type_missile = 1 if target_type == 'Missile' else 0
    
    # Create a DataFrame from the form input
    input_data = pd.DataFrame([{
        "Weapon Speed (m/s)": weapon_speed,
        "Target Distance (m)": target_distance,
        "Weapon Type_Missile": weapon_type_missile,
        "Weapon Type_Shotgun": weapon_type_shotgun,
        "Weapon Type_Cannon": weapon_type_cannon,
        "Weapon Type_Drone": weapon_type_drone,
        "Weapon Type_Interceptor": weapon_type_interceptor,
        "Target Type_Aircraft": target_type_aircraft,
        "Target Type_Bunker": target_type_bunker,
        "Target Type_Infantry": target_type_infantry,
        "Target Type_Missile": target_type_missile
    }], columns=feature_columns)

    # Debugging: Print input data columns to check the match with model features
    print("Input data columns:", input_data.columns)
    print("Model feature columns:", feature_columns)

    # Ensure the input features match the expected ones (you can remove this after confirming)
    if set(input_data.columns) != set(feature_columns):
        raise ValueError("Feature columns don't match. Ensure you are passing the correct ones.")

    # Predict assigned time (regression model)
    predicted_time = time_model.predict(input_data)[0]

    # Predict success (classification model)
    predicted_success = success_model.predict(input_data)[0]

    # Prepare result data
    result = {
        "predicted_assigned_time": predicted_time,
        "predicted_success": "Success" if predicted_success == 1 else "Failure"
    }

    return render_template("result.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)

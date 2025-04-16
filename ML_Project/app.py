from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

app = Flask(__name__)

# Global variables for models
water_model = None
water_scaler = None
water_means = None
potable_means = None
motor_model = None
motor_scaler = None
motor_means = None
healthy_means = None

def initialize_models():
    """Initialize and train all machine learning models"""
    global water_model, water_scaler, water_means, potable_means
    global motor_model, motor_scaler, motor_means, healthy_means
    
    print("Initializing models...")
    
    try:
        # Load and prepare water quality data
        df = pd.read_csv("water_potability.csv")
        imputer = SimpleImputer(strategy="mean")
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
        X = df_imputed.drop("Potability", axis=1)
        y = df_imputed["Potability"]
        
        # Initialize water model
        water_scaler = StandardScaler()
        X_scaled = water_scaler.fit_transform(X)
        water_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        water_model.fit(X_scaled, y)
        water_means = df_imputed.mean()
        potable_means = df_imputed[df_imputed["Potability"] == 1].mean()
        
        # Initialize motor model (using same data structure for demo)
        motor_scaler = StandardScaler()
        motor_scaler.fit(X)
        motor_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        motor_model.fit(X_scaled, y)
        motor_means = df_imputed.mean()
        healthy_means = df_imputed[df_imputed["Potability"] == 1].mean()
        
        print("Models initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing models: {str(e)}")
        return False

# Initialize models before first request
@app.before_request
def before_first_request():
    if water_model is None:
        initialize_models()

@app.route('/')
def home():
    return render_template('water.html', active='water')

@app.route('/water')
def water():
    return render_template('water.html', active='water')

@app.route('/motor')
def motor():
    return render_template('motor.html', active='motor')

@app.route('/about')
def about():
    return render_template('about.html', active='about')

@app.route('/predict_water', methods=['POST'])
def predict_water():
    if water_model is None:
        return jsonify({'error': 'Models not initialized'}), 500
    
    try:
        data = request.json
        
        # Construct full feature input
        user_input = [
            float(data['ph']),
            float(data['hardness']),
            water_means["Solids"],
            water_means["Chloramines"],
            float(data['sulfate']),
            water_means["Conductivity"],
            water_means["Organic_carbon"],
            water_means["Trihalomethanes"],
            float(data['turbidity'])
        ]
        
        input_scaled = water_scaler.transform([user_input])
        prediction = water_model.predict(input_scaled)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'user_vals': [
                float(data['ph']),
                float(data['hardness']),
                float(data['sulfate']),
                float(data['turbidity'])
            ],
            'ideal_vals': [
                potable_means["ph"],
                potable_means["Hardness"],
                potable_means["Sulfate"],
                potable_means["Turbidity"]
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_motor', methods=['POST'])
def predict_motor():
    if motor_model is None:
        return jsonify({'error': 'Models not initialized'}), 500
    
    try:
        data = request.json
        
        # For demo purposes, using same structure as water
        user_input = [
            float(data['temp']),
            float(data['vibration']),
            motor_means["Solids"],
            motor_means["Chloramines"],
            float(data['voltage']),
            motor_means["Conductivity"],
            motor_means["Organic_carbon"],
            motor_means["Trihalomethanes"],
            float(data['noise'])
        ]
        
        input_scaled = motor_scaler.transform([user_input])
        prediction = motor_model.predict(input_scaled)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'user_vals': [
                float(data['temp']),
                float(data['vibration']),
                float(data['voltage']),
                float(data['noise'])
            ],
            'ideal_vals': [
                healthy_means["ph"],
                healthy_means["Hardness"],
                healthy_means["Sulfate"],
                healthy_means["Turbidity"]
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# CLI command for manual initialization
@app.cli.command()
def init_models():
    """Initialize the machine learning models"""
    if initialize_models():
        print("Models initialized successfully")
    else:
        print("Failed to initialize models")

if __name__ == '__main__':
    # Initialize models when running directly
    if initialize_models():
        app.run(debug=True)
    else:
        print("Failed to start application due to model initialization error")
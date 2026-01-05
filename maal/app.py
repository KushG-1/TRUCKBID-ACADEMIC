
# backend/app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS
from datetime import datetime
from ktm_distances import get_distance
import random

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Load trained model and encoders
model = joblib.load('models/truck_pricing_model.pkl')
encoders = joblib.load('models/label_encoders.pkl')

def get_distance_km(pickup, dropoff):
    """Get distance between two Kathmandu areas"""
    return get_distance(pickup, dropoff)

def get_traffic_level(pickup, dropoff):
    """Get realistic traffic level for Kathmandu route"""
    # Always congested routes
    always_heavy = [
        ("Kalanki", "Koteshwor"),
        ("Baneshwor", "Koteshwor"),
        ("New Road", "Baneshwor"),
        ("Kalanki", "Swayambhu")
    ]
    
    route = (pickup, dropoff)
    reverse_route = (dropoff, pickup)
    
    if route in always_heavy or reverse_route in always_heavy:
        return random.choice(["Heavy", "Very Heavy"])
    
    # Normal traffic
    return random.choice(["Light", "Medium", "Heavy"])

def get_time_of_day(hour):
    """Determine time of day from hour"""
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

def is_peak_hour(hour):
    """Check if current hour is peak hour (7-9 AM, 5-7 PM)"""
    return 1 if hour in [7, 8, 9, 17, 18, 19] else 0

def prepare_features(distance_km, truck_type, traffic_level, time_of_day, peak_hours):
    """Prepare features for model prediction"""
    # Encode categorical features
    features = {
        'distance_km': distance_km,
        'truck_category': encoders['truck_category'].transform([truck_type])[0],
        'traffic_level': encoders['traffic_level'].transform([traffic_level])[0],
        'time_of_day': encoders['time_of_day'].transform([time_of_day])[0],
        'is_peak_hour': peak_hours
    }
    return list(features.values())

def calculate_breakdown(distance, truck_type, traffic_level, peak_hours):
    """Calculate price breakdown"""
    # Simple breakdown
    base_rate = {'SMALL': 18, 'MEDIUM': 25, 'LARGE': 35}[truck_type]
    base_price = distance * base_rate
    
    traffic_multipliers = {
        "Light": 1.0,
        "Medium": 1.1,
        "Heavy": 1.3,
        "Very Heavy": 1.5
    }
    
    traffic_factor = traffic_multipliers.get(traffic_level, 1.1)
    peak_factor = 1.2 if peak_hours else 1.0
    
    final_price = base_price * traffic_factor * peak_factor
    
    return {
        'base_price': round(base_price, 2),
        'traffic_adjustment': round(base_price * (traffic_factor - 1), 2),
        'peak_adjustment': round(base_price * traffic_factor * (peak_factor - 1), 2),
        'total': round(final_price, 2)
    }

@app.route('/api/predict', methods=['POST'])
def predict_price():
    try:
        # Get data from request
        data = request.get_json()
        
        # Step 1: Get distance
        distance = get_distance_km(data['pickup'], data['dropoff'])
        
        # Step 2: Get current traffic and time data
        current_time = datetime.now()
        traffic_level = get_traffic_level(data['pickup'], data['dropoff'])
        time_of_day = get_time_of_day(current_time.hour)
        peak_hours = is_peak_hour(current_time.hour)
        
        # Step 3: Prepare features for model
        features = prepare_features(
            distance_km=distance,
            truck_type=data['truck_type'],
            traffic_level=traffic_level,
            time_of_day=time_of_day,
            peak_hours=peak_hours
        )
        
        # Step 4: Make prediction
        predicted_price = model.predict([features])[0]
        
        # Step 5: Add random variation (±10%)
        variation = np.random.uniform(0.9, 1.1)
        final_price = round(predicted_price * variation, 2)
        
        # Step 6: Calculate confidence range
        lower_bound = round(final_price * 0.85, 2)
        upper_bound = round(final_price * 1.15, 2)
        
        return jsonify({
            'predicted_price': final_price,
            'price_range': {
                'min': lower_bound,
                'max': upper_bound
            },
            'distance_km': distance,
            'traffic_level': traffic_level,
            'time_of_day': time_of_day,
            'peak_hours': peak_hours,
            'breakdown': calculate_breakdown(distance, data['truck_type'], traffic_level, peak_hours)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False)
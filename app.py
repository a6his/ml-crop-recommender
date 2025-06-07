from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
from model_loader_enhanced import get_model_loader
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key

# Load crop recommendations data
def load_crop_recommendations():
    """Load crop min/max values from CSV file"""
    try:
        csv_path = 'crop_min_max_values.csv'  # Adjust path as needed
        df = pd.read_csv(csv_path)
        # Convert to dictionary with crop name as key
        recommendations = {}
        for _, row in df.iterrows():
            crop = row['Crop'].lower().strip()
            # Handle missing values by using 0 as default
            recommendations[crop] = {
                'N': {'min': row.get('N_min', 0), 'max': row.get('N_max', 0)},
                'P': {'min': row.get('P_min', 0), 'max': row.get('P_max', 0)},
                'K': {'min': row.get('K_min', 0), 'max': row.get('K_max', 0)},
                'temperature': {'min': row.get('temperature_min', 0), 'max': row.get('temperature_max', 0)},
                'humidity': {'min': row.get('humidity_min', 0), 'max': row.get('humidity_max', 0)},
                'ph': {'min': row.get('ph_min', 0), 'max': row.get('ph_max', 0)},
                'rainfall': {'min': row.get('rainfall_min', 0), 'max': row.get('rainfall_max', 0)}
            }
        
        print(f"Loaded recommendations for crops: {list(recommendations.keys())}")  # Debug line
        return recommendations
    except Exception as e:
        print(f"Error loading crop recommendations: {e}")
        return {}

# Initialize model loader and crop recommendations
try:
    ml = get_model_loader()
    print(f"Application started with model: {ml.model_name}")
except Exception as e:
    print(f"Error initializing model: {e}")
    ml = None

# Load crop recommendations
crop_recommendations = load_crop_recommendations()

def get_crop_recommendations(predicted_crop):
    """Get recommended ranges for a specific crop"""
    if not predicted_crop:
        return None
        
    crop_lower = predicted_crop.lower().strip()
    print(f"Debug: Looking for crop '{crop_lower}' in recommendations")  # Debug line
    
    # Try exact match first
    if crop_lower in crop_recommendations:
        return crop_recommendations[crop_lower]
    
    # Try partial matching
    for crop_key in crop_recommendations.keys():
        if crop_lower in crop_key or crop_key in crop_lower:
            print(f"Debug: Found partial match '{crop_key}' for '{crop_lower}'")
            return crop_recommendations[crop_key]
    
    print(f"Debug: No match found for '{crop_lower}'")
    return None

def format_recommendations(recommendations):
    """Format recommendations for display"""
    if not recommendations:
        return None
    
    formatted = {}
    for param, values in recommendations.items():
        if param == 'temperature':
            formatted[param] = f"{values['min']}-{values['max']}°C"
        elif param == 'humidity':
            formatted[param] = f"{values['min']}-{values['max']}%"
        elif param == 'rainfall':
            formatted[param] = f"{values['min']}-{values['max']} mm"
        elif param in ['N', 'P', 'K']:
            formatted[param] = f"{values['min']}-{values['max']} kg/ha"
        elif param == 'ph':
            formatted[param] = f"{values['min']}-{values['max']}"
        else:
            formatted[param] = f"{values['min']}-{values['max']}"
    
    return formatted

@app.route('/')
def index():
    """Home page with input form"""
    return render_template('index.html', model_name=ml.model_name if ml else "Model not loaded")

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if not ml:
            flash('Model not loaded. Please check your model files.', 'error')
            return redirect(url_for('index'))

        # Get form data
        input_data = {
            'N': float(request.form.get('nitrogen', 0)),
            'P': float(request.form.get('phosphorus', 0)),
            'K': float(request.form.get('potassium', 0)),
            'temperature': float(request.form.get('temperature', 0)),
            'humidity': float(request.form.get('humidity', 0)),
            'ph': float(request.form.get('ph', 0)),
            'rainfall': float(request.form.get('rainfall', 0))
        }

        # Validate input ranges
        validation_errors = validate_input(input_data)
        if validation_errors:
            for error in validation_errors:
                flash(error, 'error')
            return redirect(url_for('index'))

        # Make prediction
        result = ml.predict(input_data)
        
        # Get recommendations for the predicted crop
        # Check different possible keys for the predicted crop
        predicted_crop = (result.get('prediction') or 
                         result.get('predicted_crop') or 
                         result.get('crop', '')).lower()
        
        print(f"Debug: Predicted crop = '{predicted_crop}'")  # Debug line
        print(f"Debug: Available crops = {list(crop_recommendations.keys())}")  # Debug line
        
        crop_rec = get_crop_recommendations(predicted_crop)
        formatted_recommendations = format_recommendations(crop_rec)
        
        # Compare input values with recommendations
        comparison = None
        if crop_rec:
            comparison = {}
            for param in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
                input_val = input_data[param]
                min_val = crop_rec[param]['min']
                max_val = crop_rec[param]['max']
                
                if input_val < min_val:
                    status = 'low'
                    suggestion = f"Increase to {min_val}-{max_val}"
                elif input_val > max_val:
                    status = 'high'
                    suggestion = f"Decrease to {min_val}-{max_val}"
                else:
                    status = 'optimal'
                    suggestion = "Within optimal range"
                
                comparison[param] = {
                    'current': input_val,
                    'recommended_min': min_val,
                    'recommended_max': max_val,
                    'status': status,
                    'suggestion': suggestion
                }

        return render_template('predict.html',
                             result=result,
                             input_data=input_data,
                             model_name=ml.model_name,
                             recommendations=formatted_recommendations,
                             comparison=comparison
                             )

    except ValueError as e:
        flash(f'Invalid input values: {str(e)}', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'An error occurred during prediction: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if not ml:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Validate required fields
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Validate input ranges
        validation_errors = validate_input(data)
        if validation_errors:
            return jsonify({'error': validation_errors}), 400

        # Make prediction
        result = ml.predict(data)
        result['input_data'] = data
        
        # Add recommendations to API response
        predicted_crop = result.get('prediction', '').lower()
        crop_rec = get_crop_recommendations(predicted_crop)
        if crop_rec:
            result['recommendations'] = crop_rec
            
            # Add comparison
            comparison = {}
            for param in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
                input_val = data[param]
                min_val = crop_rec[param]['min']
                max_val = crop_rec[param]['max']
                
                if input_val < min_val:
                    status = 'low'
                elif input_val > max_val:
                    status = 'high'
                else:
                    status = 'optimal'
                
                comparison[param] = {
                    'current': input_val,
                    'recommended_min': min_val,
                    'recommended_max': max_val,
                    'status': status
                }
            
            result['comparison'] = comparison

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """Display model information"""
    if not ml:
        return jsonify({'error': 'Model not loaded'}), 500

    info = {
        'model_name': ml.model_name,
        'feature_names': ml.feature_names,
        'scaler_used': ml.best_scaler is not None,
        'crops_supported': list(crop_recommendations.keys()) if crop_recommendations else []
    }

    # Try to read model comparison data
    try:
        comparison_path = os.path.join(ml.models_dir, 'model_comparison.csv')
        if os.path.exists(comparison_path):
            comparison_df = pd.read_csv(comparison_path)
            info['model_comparison'] = comparison_df.to_dict('records')
    except:
        pass

    return jsonify(info)

@app.route('/debug_crops')
def debug_crops():
    """Debug endpoint to check loaded crops"""
    return jsonify({
        'loaded_crops': list(crop_recommendations.keys()),
        'total_crops': len(crop_recommendations),
        'sample_data': {k: v for k, v in list(crop_recommendations.items())[:3]}
    })

@app.route('/crops_info')
def crops_info():
    """API endpoint to get all crop recommendations"""
    return jsonify(crop_recommendations)

def validate_input(data):
    """Validate input data ranges"""
    errors = []

    # Define reasonable ranges for each parameter
    ranges = {
        'N': (0, 300),        # Nitrogen (kg/ha)
        'P': (0, 150),        # Phosphorus (kg/ha)
        'K': (0, 300),        # Potassium (kg/ha)
        'temperature': (8, 50), # Temperature (°C)
        'humidity': (10, 100),  # Humidity (%)
        'ph': (3, 10),         # pH level
        'rainfall': (20, 300)   # Rainfall (mm)
    }

    for param, (min_val, max_val) in ranges.items():
        if param in data:
            value = data[param]
            if not (min_val <= value <= max_val):
                errors.append(f'{param.title()} must be between {min_val} and {max_val}')

    return errors

@app.route('/about')
def about():
    """About page with information about the application"""
    return render_template('about.html')

@app.context_processor
def inject_model_name():
    return {
        'model_name': ml.model_name if ml else "Model not loaded"
    }

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
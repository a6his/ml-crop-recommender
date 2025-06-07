import pandas as pd
import joblib
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import traceback

class ModelLoader:
    def __init__(self, models_dir='saved_models'):
        self.models_dir = models_dir
        self.best_model = None
        self.best_scaler = None
        self.label_encoder = None
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.model_name = None
        self.is_loaded = False
        self.error_message = None
        
        print("Initializing Enhanced ModelLoader...")
        try:
            self.load_best_model()
            if self.best_model is not None and self.label_encoder is not None:
                self.is_loaded = True
                print(f"✅ ModelLoader initialized successfully")
                print(f"   Model: {self.model_name}")
                print(f"   Crops available: {len(self.label_encoder.classes_)}")
                self.test_prediction()
            else:
                self.error_message = "Model or label encoder not loaded"
                print(f"❌ {self.error_message}")
        except Exception as e:
            self.error_message = str(e)
            print(f"❌ ModelLoader initialization failed: {e}")
            traceback.print_exc()
    
    def load_best_model(self):
        """Load the best model with proper label encoder"""
        if not os.path.exists(self.models_dir):
            raise FileNotFoundError(f"Models directory '{self.models_dir}' does not exist")
        
        # Load model comparison to find best model
        comparison_path = os.path.join(self.models_dir, 'model_comparison.csv')
        if os.path.exists(comparison_path):
            comparison_df = pd.read_csv(comparison_path)
            best_model_name = comparison_df.iloc[0]['model_name']
            print(f"Best model from comparison: {best_model_name}")
            
            # Map to file prefix
            name_to_prefix = {
                'K-Nearest Neighbors': 'knn',
                'Logistic Regression': 'logistic_regression',
                'Random Forest': 'random_forest',
                'Decision Tree': 'decision_tree',
                'Support Vector Machine': 'svm',
                'Gaussian Naive Bayes': 'naive_bayes',
                'XGBoost': 'xgboost',
                'MLP Classifier': 'mlp'
            }
            
            model_prefix = name_to_prefix.get(best_model_name, 'knn')
            self.model_name = best_model_name
        else:
            # Default to KNN
            model_prefix = 'knn'
            self.model_name = 'K-Nearest Neighbors'
            print("No comparison file, defaulting to KNN")
        
        # Load model
        model_path = os.path.join(self.models_dir, f'{model_prefix}_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.best_model = joblib.load(model_path)
        print(f"✅ Model loaded: {type(self.best_model).__name__}")
        
        # Load scaler
        scaler_path = os.path.join(self.models_dir, f'{model_prefix}_scaler.pkl')
        if os.path.exists(scaler_path):
            self.best_scaler = joblib.load(scaler_path)
            print(f"✅ Scaler loaded: {type(self.best_scaler).__name__}")
        
        # Load label encoder (CRITICAL for proper crop names)
        self.load_label_encoder()
    
    def load_label_encoder(self):
        """Load label encoder with verification"""
        le_path = os.path.join(self.models_dir, 'label_encoder.pkl')
        
        if os.path.exists(le_path):
            try:
                self.label_encoder = joblib.load(le_path)
                print(f"✅ Label encoder loaded with {len(self.label_encoder.classes_)} classes:")
                for i, crop in enumerate(self.label_encoder.classes_):
                    print(f"      {i}: {crop}")
                return
            except Exception as e:
                print(f"❌ Error loading label encoder: {e}")
        
        # If no label encoder, create from original data
        print("Creating label encoder from original data...")
        try:
            df = pd.read_csv('crop_data.csv')
            if 'crop' in df.columns:
                unique_crops = sorted(df['crop'].unique())
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(unique_crops)
                
                # Save for future use
                joblib.dump(self.label_encoder, le_path)
                print(f"✅ Created and saved label encoder with crops: {unique_crops}")
                return
        except Exception as e:
            print(f"Could not create from original data: {e}")
        
        # Final fallback
        print("Using fallback crop names...")
        fallback_crops = [
            'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 
            'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans',
            'lentil', 'maize', 'mango', 'mothbeans', 'mungbean',
            'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate',
            'rice', 'watermelon'
        ]
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(fallback_crops)
        print(f"✅ Fallback label encoder created with {len(fallback_crops)} crops")
    
    def test_prediction(self):
        """Test prediction to ensure everything works"""
        try:
            test_data = np.random.rand(1, 7)
            if self.best_scaler:
                test_data = self.best_scaler.transform(test_data)
            
            if self.best_model is not None:
                prediction = self.best_model.predict(test_data)[0]
                crop_name = self.convert_prediction_to_crop_name(prediction)
                print(f"✅ Test prediction: {prediction} -> {crop_name}")
            else:
                print("❌ No model loaded to perform test prediction.")
        except Exception as e:
            print(f"❌ Test prediction failed: {e}")
            raise
    
    def predict(self, input_data):
        """Make prediction with proper crop name conversion"""
        if not self.is_loaded:
            raise Exception(f"Model not loaded properly. Error: {self.error_message}")
        
        try:
            print(f"Making prediction with input: {input_data}")
            
            # Convert input to DataFrame
            if isinstance(input_data, dict):
                for feature in self.feature_names:
                    if feature not in input_data:
                        raise ValueError(f"Missing required feature: {feature}")
                
                input_df = pd.DataFrame([input_data])[self.feature_names]
            else:
                input_df = pd.DataFrame(input_data, columns=self.feature_names)
            
            # Convert to numpy array
            input_array = input_df.values
            
            # Apply scaling if available
            if self.best_scaler is not None:
                input_scaled = self.best_scaler.transform(input_array)
            else:
                input_scaled = input_array
            
            # Make prediction
            prediction = self.best_model.predict(input_scaled)[0]
            print(f"Raw prediction (numeric): {prediction}")
            
            # Convert to crop name
            crop_name = self.convert_prediction_to_crop_name(prediction)
            print(f"Converted to crop name: {crop_name}")
            
            # Get probabilities if available
            probabilities = None
            confidence = None
            if hasattr(self.best_model, 'predict_proba'):
                probabilities = self.best_model.predict_proba(input_scaled)[0]
                confidence = float(np.max(probabilities))
                print(f"Confidence: {confidence:.3f}")
            
            result = {
                'predicted_crop': crop_name,
                'confidence': confidence,
                'model_used': self.model_name,
                'raw_prediction': int(prediction),
                'all_probabilities': probabilities.tolist() if probabilities is not None else None
            }
            
            return result
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            traceback.print_exc()
            raise Exception(f"Prediction failed: {e}")
    
    def convert_prediction_to_crop_name(self, prediction_value):
        """Convert numerical prediction to actual crop name"""
        try:
            # Ensure prediction is an integer
            pred_int = int(prediction_value)
            
            # Check if label_encoder is loaded
            if self.label_encoder is None or not hasattr(self.label_encoder, "classes_"):
                print(f"Warning: label_encoder is not loaded or invalid.")
                return f"Unknown_Crop_{pred_int}"
            
            # Check if prediction is within valid range
            if pred_int < 0 or pred_int >= len(self.label_encoder.classes_):
                print(f"Warning: Prediction {pred_int} is out of range (0-{len(self.label_encoder.classes_)-1})")
                return f"Unknown_Crop_{pred_int}"
            
            # Convert using label encoder
            crop_name = self.label_encoder.inverse_transform([pred_int])[0]
            return crop_name
            
        except Exception as e:
            print(f"Error converting prediction {prediction_value} to crop name: {e}")
            return f"Crop_{prediction_value}"
    
    def get_all_crops(self):
        """Get list of all available crops"""
        if self.label_encoder:
            return list(self.label_encoder.classes_)
        return []

# Global model loader
model_loader = None

def get_model_loader():
    global model_loader
    if model_loader is None:
        model_loader = ModelLoader()
    return model_loader

def test_model_loader():
    """Test the enhanced model loader"""
    print("=== TESTING ENHANCED MODEL LOADER ===")
    
    try:
        ml = ModelLoader()
        
        if ml.is_loaded:
            print("✅ Model loader initialized successfully")
            print(f"Available crops: {ml.get_all_crops()}")
            
            # Test with realistic data
            test_cases = [
                {'N': 90, 'P': 42, 'K': 43, 'temperature': 20.8, 'humidity': 82, 'ph': 6.5, 'rainfall': 202},  # Rice-like
                {'N': 20, 'P': 125, 'K': 200, 'temperature': 20, 'humidity': 70, 'ph': 6.5, 'rainfall': 180},  # Apple-like
                {'N': 120, 'P': 60, 'K': 60, 'temperature': 25, 'humidity': 70, 'ph': 7.0, 'rainfall': 75},   # Cotton-like
            ]
            
            for i, test_data in enumerate(test_cases, 1):
                print(f"\nTest case {i}:")
                result = ml.predict(test_data)
                print(f"   Predicted crop: {result['predicted_crop']}")
                if result['confidence']:
                    print(f"   Confidence: {result['confidence']:.3f}")
            
            return True
        else:
            print(f"❌ Model loader failed: {ml.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_loader()
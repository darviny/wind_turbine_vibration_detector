#!/usr/bin/env python3
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class OneClassSVMDetector:
    def __init__(self, model_path='models/model_svm.pkl', scaler_path='models/scaler.pkl', sensitivity=0.5, threshold=-0.5):
        
        # Load model
        try:
            model_data = joblib.load(model_path)
            # Contains feature names
            # The difference between having feature names or not is significant:
            # 1. With feature names: The model knows which features correspond to which columns
            #    This ensures correct mapping between input data and model features
            # 2. Without feature names: The model assumes features are in the same order as during training
            #    This can lead to errors if the order of features changes
            # Feature names are crucial for maintaining consistency between training and prediction
            if isinstance(model_data, dict):
                self.model = model_data['model']
                self.feature_names = model_data.get('feature_names', [])
                print("Loaded model from dictionary")
                print(f"Feature names: {self.feature_names}")
            else:
                self.model = model_data
                self.feature_names = []
                print("Loaded model directly")
            print(f"Model type: {type(self.model)}")
            
            # Set sensitivity (0.0 to 1.0, higher = less sensitive)
            self.sensitivity = max(0.0, min(1.0, sensitivity))
            print(f"SVM sensitivity set to {self.sensitivity}")
            
            # Set threshold directly
            self.threshold = threshold
            print(f"Anomaly threshold set to {self.threshold}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
        
        # Load scaler
        try:
            scaler_data = joblib.load(scaler_path)
            # Contains feature names
            if isinstance(scaler_data, dict):
                self.scaler = scaler_data['scaler']
                self.scaler_feature_names = scaler_data.get('feature_names', [])
                print("Loaded scaler from dictionary")
                print(f"Scaler feature names: {self.scaler_feature_names}")
            else:
                self.scaler = scaler_data
                self.scaler_feature_names = []
                print("Loaded scaler directly")
            print("Successfully loaded scaler")
        except (FileNotFoundError, IOError):
            print(f"Scaler file {scaler_path} not found. Using identity scaling.")
            self.scaler = None
                
    def predict(self, features):
        if features is None or self.model is None:
            return 0.0
            
        # 1. np.array(features): Converts the input features to a NumPy array if it isn't already
        # 2. .reshape(1, -1): Reshapes the array to have 1 row and automatically determines the number of columns
        #    The -1 tells NumPy to calculate the number of columns based on the total number of elements
        # This reshaping is necessary because scikit-learn models expect 2D arrays with shape (n_samples, n_features)
        # Even for a single sample, we need a 2D array with shape (1, n_features)
        
        # We only use 1 row because the predict method is designed to process one sample at a time
        # This is appropriate for real-time anomaly detection where we're checking each new sensor reading
        # as it comes in, rather than batch processing multiple samples
        features = np.array(features).reshape(1, -1)

        # If a scaler is available, it means we trained the model with feature scaling
        # This is important for consistent performance and preventing features with larger values from dominating   
        if self.scaler is not None:
            try:
                # Convert features to DataFrame with feature names if available
                if hasattr(self, 'scaler_feature_names') and self.scaler_feature_names:
                    print(f"Using scaler feature names: {self.scaler_feature_names}")
                    features_df = pd.DataFrame(features, columns=self.scaler_feature_names)
                    features = self.scaler.transform(features_df)
                else:
                    print("No feature names available, using direct transformation")
                    features = self.scaler.transform(features)
            except Exception as e:
                print(f"Error in scaling: {e}")
                return 0.0
        
        try:
            # Try to use decision_function if available
            if hasattr(self.model, 'decision_function'):

                # 1. The decision_function returns the signed distance to the separating hyperplane
                # 2. Positive scores indicate the sample is inside the decision boundary (normal)
                # 3. Negative scores indicate the sample is outside the decision boundary (anomaly)
                # 4. The magnitude of the score indicates how far the sample is from the boundary
                # 5. A score of 0 means the sample is exactly on the decision boundary
                # 6. The threshold parameter determines the cutoff between normal and anomalous
                #    - Scores below the threshold are considered anomalies
                #    - Scores above the threshold are considered normal
                score = self.model.decision_function(features)[0]
                print(f"SVM score: {score}")
                print(f"Threshold: {self.threshold}")
                
                return float(score)
            else:
                # If no decision_function, use predict and return -1 for anomalies
                pred = self.model.predict(features)[0]
                score = -1.0 if pred == -1 else 1.0
                return float(score)
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 0.0
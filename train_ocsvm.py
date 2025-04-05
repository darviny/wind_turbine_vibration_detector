#!/usr/bin/env python3
"""
Usage:
    python train_ocsvm.py input_file.csv
"""
import joblib
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import sensor

def load_data(file_path):
    data = pd.read_csv(file_path)
    print(f"Loaded data shape: {data.shape}")
    return data


def get_features(data):
    features = []
    feature_names = sensor.get_feature_names()
    
    for col_name in feature_names:
        if col_name in data.columns:
            features.append(data[col_name].values)
        else:
            print(f"Warning: Column {col_name} not found in data")
    
    features = np.array(features).T
    
    print(f"Total features extracted: {features.shape[1]}")
    print(f"Number of samples: {features.shape[0]}")
    print(f"Feature names: {feature_names}")
    return features, feature_names

def train_and_save_model(features, feature_names, model_path, scaler_path):
    try:
        # Convert features to DataFrame with feature names
        features_df = pd.DataFrame(features, columns=feature_names)
        print("\nFeatures DataFrame shape:", features_df.shape)

        # Without proper scaling, features with larger values could dominate the model's decision boundary.
        scaler = StandardScaler()
        
        # The fit method:
        # 1. Computes statistical parameters (mean and standard deviation) for each feature
        # 2. Learns the data distribution from the training data
        # 3. Stores these statistics internally in the scaler object
        # 4. Does NOT transform the data yet - that happens in the transform() method
        # 5. Processes each feature independently to ensure proper scaling
        scaler.fit(features_df)
        
        # The transform method:
        # 1. Uses the statistics learned during fit() to standardize the data
        # 2. Transforms each feature to have zero mean and unit variance
        # 3. Applies the formula: z = (x - mean) / std_dev
        # 4. Returns the standardized features ready for model training
        scaled_features = scaler.transform(features_df)
        
        # Train model
        # nu=0.1: Controls the upper bound on the fraction of training errors and lower bound on fraction of support vectors
        #         A value of 0.1 means the model expects at most 10% of training data to be outliers
        # kernel='rbf': Uses Radial Basis Function (Gaussian) kernel
        #               Effective for capturing non-linear relationships in the data
        #               Well-suited for anomaly detection with non-linear boundaries
        # gamma='scale': Automatically sets gamma to 1/(n_features * X.var())
        #                Adapts to the scale of input features
        #                Works well with standardized data
        model = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
        model.fit(scaled_features)
        
        # Save model with feature names
        model_data = {
            'model': model,
            'feature_names': feature_names
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
        
        # Save scaler with feature names
        scaler_data = {
            'scaler': scaler,
            'feature_names': feature_names
        }
        joblib.dump(scaler_data, scaler_path)
        print(f"Scaler saved to {scaler_path}")
        
    except Exception as e:
        print(f"Error in training/saving: {e}")
        sys.exit(1)

def main():        
    input_file = sys.argv[1]
    output_model = sys.argv[2]
    output_scaler = sys.argv[3]
    data = load_data(input_file)
    features, feature_names = get_features(data)
    train_and_save_model(features, feature_names, output_model, output_scaler)
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 
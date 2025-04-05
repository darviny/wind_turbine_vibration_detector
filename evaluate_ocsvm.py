#!/usr/bin/env python3
"""
Usage:
    python evaluate_ocsvm_simple.py <model_file> <scaler_file> <normal_data.csv> <anomaly1.csv> <anomaly2.csv>
"""

import pandas as pd
import numpy as np
import joblib
import sys
from sensor import get_feature_names

def main():
    model_file = sys.argv[1]
    scaler_file = sys.argv[2]
    normal_file = sys.argv[3]
    anomaly1_file = sys.argv[4]
    anomaly2_file = sys.argv[5]
    
    # Load model and scaler
    model_data = joblib.load(model_file)
    scaler_data = joblib.load(scaler_file)
    model = model_data['model']
    scaler = scaler_data['scaler']
    
    feature_cols = get_feature_names()
    
    # Process normal data
    normal_df = pd.read_csv(normal_file)    
    normal_df = normal_df.dropna()
    normal_features = normal_df[feature_cols]
    normal_scaled = scaler.transform(normal_features)

    # The predict() method of One-Class SVM returns:
        #   +1 for samples that are classified as "normal" (inside the decision boundary)
        #   -1 for samples that are classified as "anomaly" (outside the decision boundary)
        # For anomaly data, we expect most samples to be classified as -1 (anomaly)
        # If many samples are classified as +1, it means the model is missing anomalies
    normal_predictions = model.predict(normal_scaled)
    print(f"Loaded {len(normal_df)} normal samples")
    
    anomaly_files = {
        "Anomaly 1": anomaly1_file,
        "Anomaly 2": anomaly2_file
    }
    
    anomaly_dfs = []
    anomaly_predictions = []
    
    for name, filepath in anomaly_files.items():
        anomaly_df = pd.read_csv(filepath)
        anomaly_df = anomaly_df.dropna()
        anomaly_features = anomaly_df[feature_cols]
        anomaly_scaled = scaler.transform(anomaly_features)
        predictions = model.predict(anomaly_scaled)
        anomaly_dfs.append(anomaly_df)
        anomaly_predictions.append(predictions)
        print(f"Loaded {len(anomaly_df)} {name} samples")
    
    all_anomaly_predictions = np.concatenate(anomaly_predictions)
    
    # Calculate metrics for normal data
    # Count how many normal samples were correctly classified
    n_normal_correct = sum(normal_predictions == 1)
    total_normal = len(normal_predictions)
    
    # Count how many anomaly samples were correctly classified
    n_anomaly_correct = sum(all_anomaly_predictions == -1)
    total_anomaly = len(all_anomaly_predictions)
    total_samples = total_normal + total_anomaly
    correct_predictions = n_normal_correct + n_anomaly_correct
    accuracy = correct_predictions / total_samples
    
    print(f"Total samples: {total_samples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Overall accuracy: {accuracy:.3f}")

    return 0

if __name__ == "__main__":
    exit(main()) 
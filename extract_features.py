#!/usr/bin/env python3
"""
USAGE:
    python extract_features_using_sensor.py <input_file> [output_file] [sampling_rate] [window_size]
"""

import pandas as pd
import sys
from sensor import SensorBuffer, get_feature_names

def main():
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    sampling_rate = int(sys.argv[3])
    window_size = float(sys.argv[4])
    
    print("Starting feature extraction using SensorBuffer...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Window size: {window_size} seconds")
    
    try:
        # Load sensor data
        columns = [
            'timestamp',
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'temperature'
        ]
        
        df = pd.read_csv(input_file, names=columns)
        print(f"Loaded {len(df)} rows of sensor data")
        
        # Drop missing values
        missing = df.isnull().sum()
        if missing.any():
            print("\nDropping rows with missing values:")
            print(missing[missing > 0])
            df = df.dropna()
            print(f"Remaining rows: {len(df)}")
        
        # Create a SensorBuffer instance
        buffer = SensorBuffer(window_size=window_size)
        
        # Process data row by row
        all_features = []
        window_count = 0
        
        for _, row in df.iterrows():
            # Create sensor data dictionary
            sensor_data = {
                'accel_x': row['accel_x'],
                'accel_y': row['accel_y'],
                'accel_z': row['accel_z'],
                'gyro_x': row['gyro_x'],
                'gyro_y': row['gyro_y'],
                'gyro_z': row['gyro_z'],
                'temp': row['temperature']
            }
            
            # Add reading to buffer
            features = buffer.add_reading(sensor_data, timestamp=pd.to_datetime(row['timestamp']))
            
            # If features were extracted, add them to the list
            if features is not False:
                all_features.append(features)
                window_count += 1
                
                if window_count % 10 == 0:
                    print(f"Processed {window_count} windows...")
        
        print(f"\nTotal windows processed: {window_count}")
        
        if window_count == 0:
            print("No windows were processed. Check your data and parameters.")
            return 1
        
        # Get feature names
        feature_names = get_feature_names()
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features, columns=feature_names)
        print(f"Feature matrix shape: {features_df.shape}")
        
        # Save features
        features_df.to_csv(output_file, index=False)
        print(f"\nFeatures saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
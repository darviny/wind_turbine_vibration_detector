from datetime import datetime
import numpy as np
import csv
import os


def log_sensor_data_to_csv(sensor_data, timestamp=None, filename='data/sensor_data.csv'):
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    data_row = [timestamp, 
                sensor_data['accel_x'], sensor_data['accel_y'], sensor_data['accel_z'],
                sensor_data['gyro_x'], sensor_data['gyro_y'], sensor_data['gyro_z'],
                sensor_data['temp']]
    
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if not file_exists:
            csv_writer.writerow(['timestamp', 
                               'accel_x', 'accel_y', 'accel_z',
                               'gyro_x', 'gyro_y', 'gyro_z',
                               'temperature'])
        csv_writer.writerow(data_row)
    return True


def get_feature_names():
    """Return the list of feature names in the correct order."""
    feature_names = []
    sensor_names = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    for sensor_name in sensor_names:
        feature_names.extend([
            f'{sensor_name}_mean',
            f'{sensor_name}_std',
        ])
    
    return feature_names


class SensorBuffer:
    def __init__(self, window_size):
        self.window_size = window_size
        self.start_time = None
        self.accel_x = []
        self.accel_y = []
        self.accel_z = []
        self.gyro_x = []
        self.gyro_y = []
        self.gyro_z = []
        self.last_window = None
        self.last_features = None

    def add_reading(self, sensor_data, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        
        if self.start_time is None:
            self.start_time = timestamp
        
        # Add new reading to buffer
        self.accel_x.append(sensor_data['accel_x'])
        self.accel_y.append(sensor_data['accel_y'])
        self.accel_z.append(sensor_data['accel_z'])
        self.gyro_x.append(sensor_data['gyro_x'])
        self.gyro_y.append(sensor_data['gyro_y'])
        self.gyro_z.append(sensor_data['gyro_z'])
        
        print(f"Added reading. Buffer lengths: accel_x={len(self.accel_x)}, accel_y={len(self.accel_y)}, accel_z={len(self.accel_z)}, "
              f"gyro_x={len(self.gyro_x)}, gyro_y={len(self.gyro_y)}, gyro_z={len(self.gyro_z)}")
        
        window_complete = (timestamp - self.start_time).total_seconds() >= self.window_size
        
        if window_complete:
            print("Window duration elapsed, processing...")
            
            window, features = self._process_window()
            self.start_time = timestamp
            self.last_window = window
            self.last_features = features
            
            if features is not None:
                self.accel_x = []
                self.accel_y = []
                self.accel_z = []
                self.gyro_x = []
                self.gyro_y = []
                self.gyro_z = []
                return features
            else:
                return False
        
        return False

    # Extract features
    def _process_window(self):
        if len(self.accel_x) == 0:
            print("No samples in buffer to process")
            return None, None
            
        print("Processing window with data:")
        print(f"Accel X: {self.accel_x}")
        print(f"Accel Y: {self.accel_y}")
        print(f"Accel Z: {self.accel_z}")
        print(f"Gyro X: {self.gyro_x}")
        print(f"Gyro Y: {self.gyro_y}")
        print(f"Gyro Z: {self.gyro_z}")
            
        # Use numpy array for performance of processing streaming data.
        # Shape: (6, n_samples)
        window = np.array([
            self.accel_x, self.accel_y, self.accel_z,
            self.gyro_x, self.gyro_y, self.gyro_z
        ])
        # Transpose to shape (n_samples, 6)
        window = window.T
        features = []
        
        try:
            # For each sensor
            for i in range(6):
                sensor_data = window[:, i]
                features.extend([
                    np.mean(sensor_data),
                    np.std(sensor_data) 
                ])
                print(f"Processing sensor {i} with data: {sensor_data}")
            features = np.array(features)
            
            print(f"Processed window. Total features: {len(features)}")
            return window, features
        except Exception as e:
            print(f"Error processing window: {e}")
            return None, None

    def get_latest_window(self):
        if self.last_window is not None:
            return self.last_window
        if len(self.accel_x) == 0:
            print("No samples in buffer")
            return None
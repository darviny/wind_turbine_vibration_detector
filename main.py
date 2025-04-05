#!/usr/bin/env python3
import time
import sys
import board
import adafruit_mpu6050
import numpy as np

from datetime import datetime
from lcd_alert import LCDAlert      

import sms_alert
import anomaly_detector
import sensor

# "Usage: python main.py <alerts_enabled> [sensitivity] [threshold]"
# alerts_enabled: 'true' or 'false'
# sensitivity: float between 0.0 and 1.0 (default: 0.5)
# threshold: anomaly threshold (default: -0.5)  

def format_alert(svm_score=None, sensor_data=None):
    alert = "WIND TURBINE ALERT\n"
    if svm_score is not None:
        alert += f"SVM Score: {svm_score}\n"
    if sensor_data:
        alert += f"Accel: X={sensor_data['accel_x']:.2f}, Y={sensor_data['accel_y']:.2f}, Z={sensor_data['accel_z']:.2f}\n"
        alert += f"Gyro: X={sensor_data['gyro_x']:.2f}, Y={sensor_data['gyro_y']:.2f}, Z={sensor_data['gyro_z']:.2f}\n"
    return alert

def check_anomaly(buffer, svm_detector, sensor_data):
    # Get the latest window of data
    window = buffer.get_latest_window()
    if window is None:
        print("No window data available")
        return False
        
    # Use SVM detector to check for anomalies
    svm_score = svm_detector.predict(buffer.last_features)
    print(f"SVM score: {svm_score}")
    if np.any(svm_score < svm_detector.threshold):
        print(format_alert(svm_score, sensor_data))
        return True
    else:
        print("SVM did not detect anomaly")
    return False

def main():
    alerts_enabled = True
    if sys.argv[1].lower() != 'true':
        alerts_enabled = False
    
    threshold = -2.0
    sensitivity = 0.5  # Default value
    if len(sys.argv) > 2:
        threshold = float(sys.argv[2])

    try:
        print("Starting initialization...")
        
        # Initialize components
        if alerts_enabled:
            lcd = LCDAlert()
            lcd.display_alert("Hello")
            
        i2c = board.I2C()
        sensor_device = adafruit_mpu6050.MPU6050(i2c)
        buffer = sensor.SensorBuffer(window_size=6.6) # 6.6 seconds is the periodicity of the turbine
        svm_detector = anomaly_detector.OneClassSVMDetector('models/model_svm.pkl', sensitivity, threshold)

        print("\nMonitoring!")
        print("Press Ctrl+C to stop")
        
        while True:
            # Read sensor
            accel = sensor_device.acceleration
            gyro = sensor_device.gyro
            temp = sensor_device.temperature
            timestamp = datetime.now()
            accel_x, accel_y, accel_z = accel
            gyro_x, gyro_y, gyro_z = gyro
            
            sensor_data = {
                'accel_x': accel_x, 'accel_y': accel_y, 'accel_z': accel_z,
                'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z,
                'temp': temp
            }
            
            # Log sensor data to CSV
            sensor.log_sensor_data_to_csv(sensor_data, timestamp.isoformat())
            
            # Update display
            if lcd:
                lcd.lcd.clear()
                lcd.lcd.cursor_pos = (0, 0)
                lcd.lcd.write_string(f"X:{accel_x:.1f} Y:{accel_y:.1f}")
                lcd.lcd.cursor_pos = (1, 0)
                lcd.lcd.write_string(f"Z:{accel_z:.1f}")
            
            # Check for anomalies
            if buffer.add_reading(sensor_data, timestamp):
                features = buffer.last_features
                if features is not None:
                    print("Features extracted, running anomaly detection...")
                    is_anomaly = check_anomaly(buffer, svm_detector, sensor_data)
                    if is_anomaly:
                        if lcd:
                            lcd.display_alert("ANOMALY DETECTED!")
                        if alerts_enabled:
                            alert_message = format_alert(sensor_data)
                            sms_alert.send_sms_alert('+1234567890', alert_message)
            time.sleep(0.2)  # 5 Hz
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if buffer:
            buffer._process_window()
        
        print("\nGood bye!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
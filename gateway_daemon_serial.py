import os
import sys
import time
import serial
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.preprocessing import MinMaxScaler

# --- Configurations ---
SERIAL_PORT = os.path.expanduser("~/ttyVP1")
MODEL_PATH = "models/energy_lstm.onnx"
DATA_PATH = "data/processed_energy.csv"
SPIKE_THRESHOLD = 18308.00  
SEQUENCE_LENGTH = 24

print("=== Serial Agentic Energy Gateway ===")

try:
    session = ort.InferenceSession(MODEL_PATH)
    print("ONNX Inference engine loaded successfully.")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    sys.exit(1)

# Fit global scaling parameters 
def get_fitted_scaler(data_path):
    df = pd.read_csv(data_path, index_col='Datetime', parse_dates=True)
    features = ['MW', 'Hour', 'DayOfWeek']
    scaler = MinMaxScaler()
    scaler.fit(df[features].values)
    return scaler

scaler = get_fitted_scaler(DATA_PATH)

def adjust_workloads(emergency_active):
    """
    Placeholder for physical/system mitigation actions.
    Replace these print statements with GPIO pin toggles (e.g., for an ESP32/STM32 relay)
    or local OS process throttling.
    """
    if emergency_active:
        print("Grid Stress Mitigation Active -> System running in low-power profile.")
    else:
        print("Grid Safe -> System running in standard operational profile.")

def run_serial_daemon():
    print(f"\n=== Listening for Streaming Telemetry on {SERIAL_PORT} ===")
    
    try:
        ser = serial.Serial(SERIAL_PORT, 9600, timeout=1)
    except Exception as e:
        print(f"Failed to connect to serial port: {e}")
        return

    memory_buffer = []
    
    while True:
        if ser.in_waiting > 0:
            try:
                raw_line = ser.readline().decode('utf-8').strip()
                if not raw_line:
                    continue
                
                parts = raw_line.split(',')
                if len(parts) != 3:
                    continue
                
                mw, hour, day_of_week = float(parts[0]), float(parts[1]), float(parts[2])
                current_reading = [mw, hour, day_of_week]
                
                memory_buffer.append(current_reading)
                
                if len(memory_buffer) < SEQUENCE_LENGTH:
                    print(f"Buffering Window Sequence: [{len(memory_buffer)}/{SEQUENCE_LENGTH}] points loaded...", end='\r')
                    continue

                if len(memory_buffer) > SEQUENCE_LENGTH:
                    memory_buffer.pop(0)
                
                unscaled_window = np.array(memory_buffer)
                scaled_window = scaler.transform(unscaled_window)
                input_window = np.expand_dims(scaled_window, axis=0).astype(np.float32)
                
                inputs = {session.get_inputs()[0].name: input_window}
                prediction_scaled = session.run(None, inputs)[0]
                
                dummy_row = np.zeros((1, 3))
                dummy_row[0, 0] = np.asarray(prediction_scaled).item()
                predicted_load = scaler.inverse_transform(dummy_row)[0, 0]
                
                print(f"\nLive Telemetry Signal -> Current Input MW: {mw:.1f}")
                print(f"   Predicted Next-Hour Grid Load: {predicted_load:.2f} MW (Threshold: {SPIKE_THRESHOLD} MW)")
                
                if predicted_load >= SPIKE_THRESHOLD:
                    adjust_workloads(emergency_active=True)
                else:
                    adjust_workloads(emergency_active=False)
                    
            except Exception as e:
                print(f"\nData parsing error encountered: {e}")
                
        time.sleep(0.1)

if __name__ == "__main__":
    run_serial_daemon()
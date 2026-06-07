import time
import os
import numpy as np
import pandas as pd
import onnxruntime as ort
import docker
from sklearn.preprocessing import MinMaxScaler

# Config
MODEL_PATH = "models/energy_lstm.onnx"
DATA_PATH = "data/processed_energy.csv"
SPIKE_THRESHOLD = 18308.00  
SEQUENCE_LENGTH = 24

TARGET_CONTAINERS = ["plex", "overseerr", "nextcloud"]

print("*** Initializing Agentic Energy Gateway ***")

if "DOCKER_HOST" in os.environ:
    del os.environ["DOCKER_HOST"]

try:
    docker_client = docker.from_env()
    print("✓ Successfully connected to the local Linux Docker socket.")
except Exception as e:
    print(f"Warning: Could not bind to Docker socket ({e}). Running in simulation mode.")
    docker_client = None

try:
    session = ort.InferenceSession(MODEL_PATH)
    print("✓ ONNX Inference engine loaded successfully.")
except Exception as e:
    print(f" Error loading ONNX model: {e}")
    exit(1)

def get_fitted_scaler(data_path):
    df = pd.read_csv(data_path, index_col='Datetime', parse_dates=True)
    features = ['MW', 'Hour', 'DayOfWeek']
    scaler = MinMaxScaler()
    scaler.fit(df[features].values)
    return scaler, df

scaler, raw_df = get_fitted_scaler(DATA_PATH)

def adjust_workloads(emergency_active):
    if not docker_client:
        print(f"   [Simulation] Workload Adjustment triggered. Emergency Active: {emergency_active}")
        return

    for c_name in TARGET_CONTAINERS:
        try:
            container = docker_client.containers.get(c_name)
            if emergency_active and container.status == "running":
                print(f" Grid Stress Detected! Powering down/pausing container: {c_name}")
                container.pause()
            elif not emergency_active and container.status == "paused":
                print(f" Grid Status Normal. Resuming background workloads: {c_name}")
                container.unpause()
        except docker.errors.NotFound:
            pass
        except Exception as e:
            print(f"   Error adjusting container {c_name}: {e}")


def run_gateway():
    print("\n***   Agentic Monitoring Loop Active   ***")
    
    features = ['MW', 'Hour', 'DayOfWeek']
    sample_data = raw_df[features].values
    
    for step in range(len(sample_data) - SEQUENCE_LENGTH - 10, len(sample_data) - SEQUENCE_LENGTH):
        current_time = raw_df.index[step + SEQUENCE_LENGTH]
        print(f"\n Current Timestamp: {current_time}")
        
        unscaled_window = sample_data[step:step+SEQUENCE_LENGTH]
        scaled_window = scaler.transform(unscaled_window)
        input_window = np.expand_dims(scaled_window, axis=0).astype(np.float32)
        
        inputs = {session.get_inputs()[0].name: input_window}
        prediction_scaled = session.run(None, inputs)[0]
        
        scalar_val = np.asarray(prediction_scaled).item()
        
        dummy_row = np.zeros((1, len(features)))
        dummy_row[0, 0] = scalar_val
        predicted_load = scaler.inverse_transform(dummy_row)[0, 0]
        
        print(f"   Predicted Grid Load for next hour: {predicted_load:.2f} MW (Threshold: {SPIKE_THRESHOLD} MW)")
        
        if predicted_load >= SPIKE_THRESHOLD:
            adjust_workloads(emergency_active=True)
        else:
            adjust_workloads(emergency_active=False)
            
        time.sleep(1)

if __name__ == "__main__":
    run_gateway()
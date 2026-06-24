import time
import os
import serial
import pandas as pd

SERIAL_PORT = os.path.expanduser("~/ttyVP0")
DATA_PATH = "data/processed_energy.csv"

def stream_meter_data():
    print(f"=== Mock Smart Meter Telemetry over {SERIAL_PORT} ===")
    
    try:
        ser = serial.Serial(SERIAL_PORT, 9600, timeout=1)
        print("✓ Virtual serial pipeline opened successfully.")
    except Exception as e:
        print(f"Failed to open serial port: {e}")
        return

    df = pd.read_csv(DATA_PATH, index_col='Datetime', parse_dates=True)
    print("Streaming data packets. Press Ctrl+C to stop...\n")
    
    for idx, row in df.iloc[50000:].iterrows():
        # Packet format: Megawatts,Hour,DayOfWeek
        packet = f"{row['MW']},{idx.hour},{idx.dayofweek}\n"
        ser.write(packet.encode('utf-8'))
        print(f"Transmitted Packet [{idx}]: {packet.strip()}")
        time.sleep(1.5)

if __name__ == "__main__":
    stream_meter_data()
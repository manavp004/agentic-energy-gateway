import pandas as pd
import numpy as np

def load_and_inspect(filepath):
    print("=== Phase 1: Ingesting Energy Data ===")
    # Load the CSV
    df = pd.read_csv(filepath)
    
    # Rename columns for clarity if needed (assuming standard [Datetime, MW] format)
    df.columns = ['Datetime', 'MW']
    
    # Convert Datetime string to actual datetime objects and set as index
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"Data successfully loaded. Total records: {len(df)}")
    print("\nFirst 5 rows of raw grid data:")
    print(df.head())
    
    print("\n=== Phase 2: Feature Engineering ===")
    # Extract temporal features so the LSTM understands cyclical time
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    
    # Create a target column: predicting a spike (e.g., load is in top 15% historically)
    threshold = df['MW'].quantile(0.85)
    df['Spike_Warning'] = (df['MW'] > threshold).astype(int)
    
    print(f"85th percentile grid load threshold calculated at: {threshold:.2f} MW")
    print(f"Total simulated 'Critical Grid Events' found: {df['Spike_Warning'].sum()}")
    print("\nEngineered Feature Matrix:")
    print(df[['MW', 'Hour', 'DayOfWeek', 'Spike_Warning']].head())
    
    # Save the processed data for the LSTM sprint
    df.to_csv('data/processed_energy.csv')
    print("\nProcessed dataset saved to data/processed_energy.csv")

if __name__ == "__main__":
    load_and_inspect('data/AEP_hourly.csv')
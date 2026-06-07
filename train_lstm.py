import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os

# Config
SEQUENCE_LENGTH = 24
BATCH_SIZE = 64
HIDDEN_SIZE = 50
NUM_LAYERS = 2
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using execution device: {DEVICE}")

# Data Prep
def prepare_sequences(data_path, seq_length):
    df = pd.read_csv(data_path, index_col='Datetime', parse_dates=True)

    features = ['MW', 'Hour', 'DayOfWeek']
    data = df[features].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length]) 
        y.append(scaled_data[i+seq_length, 0]) 
        
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32), scaler

X, y, scaler = prepare_sequences('data/processed_energy.csv', SEQUENCE_LENGTH)


split_idx = int(len(X) * 0.8)
train_dataset = TensorDataset(X[:split_idx], y[:split_idx])
val_dataset = TensorDataset(X[split_idx:], y[split_idx:])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


class EnergyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EnergyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.squeeze()

model = EnergyLSTM(input_size=3, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n****Starting Model Training****")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_X.size(0)
        
    train_loss /= len(train_loader.dataset)
    
# Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
    val_loss /= len(val_loader.dataset)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/energy_lstm.pth')
print("\nModel training complete. Weights saved to models/energy_lstm.pth")

dummy_input = X[0].unsqueeze(0).to(DEVICE)
torch.onnx.export(model, dummy_input, "models/energy_lstm.onnx", 
                  input_names=['input_window'], output_names=['predicted_load'],
                  dynamic_axes={'input_window': {0: 'batch_size'}, 'predicted_load': {0: 'batch_size'}})
print("Universal ONNX model exported to models/energy_lstm.onnx")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import copy
import data.embedding as embedding

class EnzymeEfficiencyFFNN(nn.Module):
    def __init__(self, input_dim=1666):
        super(EnzymeEfficiencyFFNN, self).__init__()
    
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(.2),

            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def train_FFNN(train, val, epochs=100, batch_size=64, lr=.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = embedding.concat_encoder(train)
    y_train = train["Log10_value"].to_numpy(dtype=float)
    X_val = embedding.concat_encoder(val)
    y_val = val["Log10_value"].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_T = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_T = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_T = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_T = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_T, y_train_T)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_T, y_val_T)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = EnzymeEfficiencyFFNN(input_dim=X_train.shape[1]).to(device)
    cost = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float('inf')
    best_model_weights = None
    patience = 15
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = .0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = cost(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_loader.dataset)

        
        #validation & early stop for every epoch check
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = cost(predictions, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epoch % 10 == 0 or epochs_no_improve == patience:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
        if epochs_no_improve == patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    model.load_state_dict(best_model_weights)
    return model, scaler

def get_final_val_loss(model, val_data, scaler, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    X_val = embedding.concat_encoder(val_data)
    y_val = val_data["Log10_value"].to_numpy(dtype=float)

    X_val_scaled = scaler.transform(X_val)

    X_val_T = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_T = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    val_dataset = TensorDataset(X_val_T, y_val_T)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    cost = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = cost(predictions, batch_y)
            total_loss += loss.item() * batch_X.size(0)

    return total_loss / len(val_loader.dataset)

def tune_train_FFNN(train_data, val_data):
    params = {"batch_size": 64, "lr": 0.001, "epochs": 100}
    model, scaler = train_FFNN(train_data, val_data, **params)
    val_loss = get_final_val_loss(model, val_data, scaler)
    return model, [val_loss], params, scaler
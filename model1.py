
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.distributions import Normal
import torch.nn.functional as F
import copy
from pysat.formula import CNF, WCNF
from pysat.examples.rc2 import RC2
import numpy as np

from utils_model import calculate_constr_loss_prep, calculate_constr_loss_real

class Model1(nn.Module):
    """Takes n input features -> outputs mu_1..mu_m, sigma_1..sigma_m"""
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        out = self.fc(inputs)
        return out


class MyDataset(Dataset):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.length = len(x)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx]


def train_model1(x_train, y_train, x_val, y_val, config, train_constr = None, val_constr = None, is_real = False):
    """
    Train Model1 with validation, early stopping, and best model saving.

    Args:
        x_train (torch.Tensor): Training inputs [N_train, input_dim]
        y_train (torch.Tensor): Training targets [N_train, output_dim]
        x_val (torch.Tensor): Validation inputs [N_val, input_dim]
        y_val (torch.Tensor): Validation targets [N_val, output_dim]
        config (dict): Training configuration
            {
                'batch_size': int,
                'epochs': int,
                'lr': float,
                'hidden_dim': int,
                'patience': int,
                'device': 'cuda' or 'cpu'
            }
    """
    print("BCE")
    # --- Unpack config ---
    batch_size = config.get("batch_size")
    epochs = config.get("epochs")
    lr = config.get("lr")
    hidden_dim = config.get("hidden_dim")
    patience = config.get("patience")
    device = config.get("device")

    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]

    # --- Datasets and Loaders ---
    train_loader = DataLoader(MyDataset(x_train, y_train, train_constr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MyDataset(x_val, y_val, val_constr), batch_size=batch_size, shuffle=False)

    # --- Model, Optimizer ---
    model1 = Model1(input_dim, output_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model1.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    if is_real == False:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.L1Loss()
    # --- Training Loop ---
    for epoch in range(epochs):
        model1.train()
        train_loss = 0.0
        constr_train_loss = 0.0

        for xb, yb, constr in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model1(xb)
            # if is_real == False:
            #     constraint_loss = calculate_constr_loss_prep(out, constr)
            # else:
            #     constraint_loss = calculate_constr_loss_real(out, constr)
            constraint_loss = torch.tensor(0)
            loss = criterion(out, yb) + config["constraints_weight"] * constraint_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0) 
            constr_train_loss += constraint_loss.item() * xb.size(0) 

        train_loss /= len(train_loader.dataset)
        constr_train_loss /=len(train_loader.dataset)

        # --- Validation ---
        model1.eval()
        val_loss = 0.0
        constr_val_loss = 0.0
        with torch.no_grad():
            for xb, yb, constr in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model1(xb)
                # if is_real == False:
                #     constraint_loss = calculate_constr_loss_prep(out, constr)
                # else:
                #     constraint_loss = calculate_constr_loss_real(out, constr)
                constraint_loss = torch.tensor(0)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
                constr_val_loss += constraint_loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        constr_val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} Constr {constr_train_loss} | Val Loss: {val_loss:.4f} Constr {constr_val_loss:.4f}")

        # --- Check Early Stopping ---
        if val_loss < best_val_loss - 1e-6:  # small tolerance for floating point stability
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model1.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
                break

    # --- Load Best Model Weights ---
    if best_model_state is not None:
        model1.load_state_dict(best_model_state)
        print(f"Loaded best model (Val Loss: {best_val_loss:.4f})")

    return model1

def inference_model1(model, x, config):
    """
    Run inference using a trained Model1.

    Args:
        model (Model1): Trained Model1 instance.
        x (torch.Tensor): Input features [N, input_dim].
        device (str): 'cpu' or 'cuda'.
        sample (bool): If True, draw samples from predicted Gaussians.
        num_samples (int): Number of samples to draw per input (only if sample=True).

    Returns:
        dict: {
            'mu': torch.Tensor [N, output_dim],
            'sigma': torch.Tensor [N, output_dim],
            'samples': torch.Tensor [N, num_samples, output_dim] (if sample=True)
        }
    """
    model.eval()
    x = x.to(config["device"])
    model.to(config["device"])

    with torch.no_grad():
        output = model(x)

        return output
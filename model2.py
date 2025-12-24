
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal
import torch.nn.functional as F
import copy
from model1 import inference_model1

class Model2(nn.Module):
    """
    Sequentially produces Gaussian parameters for each output o_j.
    Conditioned on Model1 outputs and partial sequence.
    """
    def __init__(self, output_dim, hidden_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(2 * output_dim - 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        return out
def shift_and_pad(arr, pad_value=-1):
    """
    For each row in arr, generate shifted versions padded with pad_value.
    Example:
        [0,1,2] -> [[-1,-1],[0,-1],[0,1]]
    Works for arbitrary number of columns.
    """
    n_rows, n_cols = arr.shape
    results = []

    for row in arr:
        for j in range(n_cols):  # up to n_cols-1 shifts
            new_row = torch.full((n_cols - 1,), pad_value, dtype=row.dtype)
            new_row[:j] = row[:j]  # take first j elements
            results.append(new_row)
    
    return torch.stack(results)

def prepare_data(model1, x, y, config):
    output_dim = y.shape[1]
    
    x_predicted_m1 = inference_model1(model1, x, config = config) 
    x_predicted_m1 = x_predicted_m1.repeat_interleave(output_dim, dim = 0)
    
    x_prefix = shift_and_pad(y)
    x = torch.cat([x_predicted_m1, x_prefix], dim=-1)
    y = torch.flatten(y).unsqueeze(1)
    return x, y

def prepare_data_test(model1, x, config):
    
    x_predicted_m1 = inference_model1(model1, x, config = config) 
    output_dim = x_predicted_m1.shape[1]
    
    x_prefix = torch.full((len(x_predicted_m1), output_dim - 1), -1)
    x = torch.cat([x_predicted_m1, x_prefix], dim=-1)
    return x, output_dim


def train_model2(model1, x_train, y_train, x_val, y_val, config, is_real):
    # Prepare input
    output_dim = y_train.shape[1]
    
    x_train, y_train = prepare_data(model1, x_train, y_train, config = config["model_1"])
    x_val, y_val = prepare_data(model1, x_val, y_val, config = config["model_1"])
    
    config = config["model_2"]
    
    batch_size = config.get("batch_size")
    epochs = config.get("epochs")
    lr = config.get("lr")
    hidden_dim = config.get("hidden_dim")
    patience = config.get("patience")
    device = config.get("device")
    
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    # --- Model, Optimizer ---
    model2 = Model2(output_dim, hidden_dim = hidden_dim).to(device)
    optimizer = torch.optim.Adam(model2.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    if is_real == False:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.L1Loss()
        
    for epoch in range(epochs):
        model2.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model2(xb)
            loss = criterion(out, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation ---
        model2.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model2(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # --- Check Early Stopping ---
        if val_loss < best_val_loss - 1e-6:  # small tolerance for floating point stability
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model2.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
                break

    # --- Load Best Model Weights ---
    if best_model_state is not None:
        model2.load_state_dict(best_model_state)
        print(f"Loaded best model (Val Loss: {best_val_loss:.4f})")

    return model2

def inference_model2(model1, model2, x, config):
    x, out_dim = prepare_data_test(model1, x, config = config["model_1"])
    config = config["model_2"]
    model2.eval()
    batch_size = config.get("batch_size")
    predicted_model2_mu = []
    predicted_model2_sigma = []
    for n_class in range(out_dim):
        mu_buf = []
        loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
        for batch in loader:
            x_b = batch[0]
            with torch.no_grad():
                mu = model2(x_b)
                mu_buf.append(mu)
        predicted_model2_mu.append(torch.flatten(torch.cat(mu_buf)))
        if n_class == out_dim - 1:
            break
        x[:, out_dim + n_class] = torch.flatten(torch.cat(mu_buf))

    return torch.stack(predicted_model2_mu, dim=0).T
    

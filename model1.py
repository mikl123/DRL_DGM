
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal
import torch.nn.functional as F
import copy

class Model1(nn.Module):
    """Takes n input features -> outputs mu_1..mu_m, sigma_1..sigma_m"""
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * output_dim)
        )

    def forward(self, inputs):
        out = self.fc(inputs)
        mu, log_sigma = torch.chunk(out, 2, dim=-1)
        sigma = F.softplus(log_sigma) + 1e-6
        return mu, sigma


def train_model1(x_train, y_train, x_val, y_val, config):
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
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    # --- Model, Optimizer ---
    model1 = Model1(input_dim, output_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model1.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    # --- Training Loop ---
    for epoch in range(epochs):
        model1.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            mu, sigma = model1(xb)
            dist = Normal(mu, sigma)
            nll_loss = -dist.log_prob(yb).mean()

            optimizer.zero_grad()
            nll_loss.backward()
            optimizer.step()
            train_loss += nll_loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation ---
        model1.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                mu, sigma = model1(xb)
                dist = Normal(mu, sigma)
                loss = -dist.log_prob(yb).mean()
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

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
        mu, sigma = model(x)

        return {"mu": mu, "sigma": sigma}

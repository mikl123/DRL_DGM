
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.distributions import Normal
import torch.nn.functional as F
import copy
from pysat.formula import CNF, WCNF
from pysat.examples.rc2 import RC2
import numpy as np

from DRL.constraints_code.parser import parse_constraints_file
from DRL.constraints_code.compute_sets_of_constraints import compute_sets_of_constraints
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
        out = self.sigmoid(self.fc(inputs))
        return out
    
def prepare_constrs_real(constr_paths):
    dict_buf = {}
    for path in set(constr_paths):
        ordering, constraints = parse_constraints_file(path)
        sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
        dict_buf[path] = (ordering, sets_of_constr)
    return dict_buf

class MyDataset(Dataset):
    def __init__(self, x, y, z, sup):
        self.x = x
        self.y = y
        self.z = z
        self.sup = sup
        print(sum(self.sup == 0))
        print(sum(self.sup == -1))
        self.length = len(x)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return (self.x[idx], 
            self.y[idx],
            self.z[idx], 
            self.sup[idx] == 0)


def train_model1(x_train, y_train, y_train_unsup, x_val, y_val, y_val_unsup, config, train_constr = None, val_constr = None, is_real = False):
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
    train_loader = DataLoader(MyDataset(x_train, y_train, train_constr, y_train_unsup), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MyDataset(x_val, y_val, val_constr, y_val_unsup), batch_size=batch_size, shuffle=False)

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
    epoch_start_constr = 30
    # --- Training Loop ---
    for epoch in range(epochs):
        model1.train()
        train_loss = 0.0
        constr_train_loss = 0.0
        epoch_start_constr -= 1
        print(epoch_start_constr)
        for xb, yb, constr, sup in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model1(xb)
            if is_real == False:
                constraint_loss = calculate_constr_loss_prep(out[~sup], constr[~sup])
            else:
                if config["constraints_weight"] == 0 or epoch_start_constr > 0:
                    constraint_loss = torch.tensor(0)
                else:
                    constraint_loss = calculate_constr_loss_real(out[~sup], xb[~sup], np.array(constr)[~sup])
            # constraint_loss = torch.tensor(0)
            
            loss = criterion(out[sup], yb[sup])
            if constraint_loss != 0:
                coef = float(loss) / float(constraint_loss) 
                loss_cumm = loss + config["constraints_weight"] * constraint_loss * coef
            else:
                loss_cumm = loss
                
            if not torch.isnan(loss_cumm):
                optimizer.zero_grad()
                loss_cumm.backward()
                optimizer.step()
                train_loss += loss.item() * sum(sup)
                constr_train_loss += constraint_loss.item() * sum(~sup)

        train_loss /= len(train_loader.dataset)
        constr_train_loss /=len(train_loader.dataset)

        # --- Validation ---
        model1.eval()
        val_loss = 0.0
        unsup_val_loss = 0.0
        constr_val_loss = 0.0
        with torch.no_grad():
            for xb, yb, constr, sup in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model1(xb)
                if is_real == False:
                    constraint_loss = calculate_constr_loss_prep(out[~sup], constr[~sup])
                else:
                    constraint_loss = calculate_constr_loss_real(out[~sup], xb[~sup], np.array(constr)[~sup])
                # constraint_loss = torch.tensor(0)
                loss = criterion(out[sup], yb[sup])
                loss_unsup = criterion(out[~sup], yb[~sup])
                # loss_unsup = 0
                
                if not torch.isnan(loss):
                    val_loss += loss.item() * sum(sup)
                if not torch.isnan(loss_unsup):
                    unsup_val_loss += loss_unsup.item() * sum(~sup)
                if not torch.isnan(constraint_loss):
                    constr_val_loss += constraint_loss.item() * sum(~sup)
        val_loss /= len(val_loader.dataset)
        constr_val_loss /= len(val_loader.dataset)
        unsup_val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} Constr {constr_train_loss} | Val Loss: {val_loss:.4f} {unsup_val_loss:.4f} Constr {constr_val_loss:.4f}")

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
from DRL.constraints_code.parser import parse_constraints_file
from DRL.constraints_code.feature_orderings import set_ordering
from DRL.constraints_code.compute_sets_of_constraints import compute_sets_of_constraints
import torch
import pandas as pd
import torch.nn as nn
from DRL.constraints_code.correct_predictions import correct_preds, check_all_constraints_sat
import torch.nn.functional as F
import random
import numpy as np
from sklearn.model_selection import train_test_split
import torch.optim as optim
import os
import csv

def seed_everything(seed: int):
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"All random seeds set to {seed}")
    
def check_vectors_against_smt2(vectors, tolerance = 0):
    """
    Check a list of vectors against SMT2 constraints and return
    the fraction of vectors that violate at least one constraint.

    Args:
        smt2_path (str): path to the SMT2 file.
        vectors (list[dict]): list of dictionaries mapping variable names to values.

    Returns:
        float: fraction of violating vectors (0.0 = all satisfy, 1.0 = all violate)
    """
    
    _, constraints = parse_constraints_file(constraint_path)
    sat_buf = 0
    for i in range(len(vectors)):
        sat = check_all_constraints_sat(vectors[i:i+1], constraints=constraints, error_raise=False, tolerance = tolerance)
        sat_buf += sat
    return 1 - (sat_buf/len(vectors))
    

# Simple MLP Generator
class Generator(nn.Module):
    def __init__(self, input_dim=16, output_dim=32, hidden_dim=64):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.model(z)

def sample(ordering_list, sets_of_constr, n=10000, batch_size=512, epochs=5):
    input_length = 20
    output_length = 10
    
    generator = Generator(input_dim=input_length, output_dim=output_length)
    optimizer = optim.Adam(generator.parameters(), lr=0.001)
    noise = torch.rand(size=(n, input_length)).float() * 5
    diversity_weight = 0.0001
    metrics = {}
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_mae = 0.0
        num_batches = 0
        
        for i in range(0, n, batch_size):
            optimizer.zero_grad()
            
            batch_idx = perm[i:i+batch_size]
            noise_batch = noise[batch_idx]
            
            generated_data = generator(noise_batch)
            generated_data = torch.sigmoid(generated_data)
            
            target = correct_preds(generated_data, ordering_list, sets_of_constr)
            target = torch.tensor(target, dtype=generated_data.dtype)
            
            col_mean = generated_data.mean(dim=0)  
            col_var = ((generated_data - col_mean)**2).mean(dim=0)  
            diversity_loss = 1.0 / (col_var + 1e-6) 
            diversity_loss = diversity_loss.mean()

            loss_f1 = F.l1_loss(generated_data, target) if epoch >= 1 else 0
            print("loss_f1", loss_f1)
            print("diversity_loss", diversity_loss)
            loss =  diversity_weight * diversity_loss + loss_f1 

            loss.backward()
            optimizer.step()
            
            epoch_mae += loss.item()
            num_batches += 1
        
        print(f"Epoch {epoch+1}, MAE: {epoch_mae/num_batches:.6f}")
    
    # Generate final sampled data
    with torch.no_grad():
        generated_data = generator(noise)
        generated_data = torch.sigmoid(generated_data)
        sampled_data = correct_preds(generated_data, ordering_list, sets_of_constr)
        loss_f1 = F.l1_loss(generated_data, sampled_data)
        batch_mean = generated_data.mean(dim=0, keepdim=True)
        diversity_loss = 1.0 / ( ((generated_data - batch_mean) ** 2).mean() + 1e-6 )
        metrics["n_violations"] = check_vectors_against_smt2(generated_data)
        metrics["loss_f1"] = loss_f1
        metrics["diversity_loss"] = diversity_loss
        
    return noise, sampled_data, metrics

import argparse

parser = argparse.ArgumentParser(description="Your script description here")

parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility"
)

parser.add_argument(
    "--constraint_path",
    type=str,
    default=None,
    help="Path to the constraint file"
)
parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    help="Path to the constraint file"
)

args = parser.parse_args()

seed = args.seed
constraint_path = args.constraint_path
save_path = "data_generated_pretrained"

seed_everything(seed)

label_ordering = "predefined"


constraints_file = constraint_path
ordering, constraints = parse_constraints_file(constraints_file)
sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
noise, sampled_data, metrics = sample(ordering, sets_of_constr)
check_all_constraints_sat(sampled_data, constraints=constraints, error_raise=True)

noise_np = noise.numpy()
predictions_np = sampled_data.numpy()

dataset_np = np.hstack([noise_np, predictions_np])

input_cols = [f"noise_{i}" for i in range(noise_np.shape[1])]
output_cols = [f"pred_{i}" for i in range(predictions_np.shape[1])]
columns = input_cols + output_cols

df = pd.DataFrame(dataset_np, columns=columns)

df_train, df_temp = train_test_split(df, test_size=0.3, random_state=seed)

df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=seed)

base_path = constraint_path.split("/")[-1][:-4] + f"_{seed}"

df_train.to_csv(f"{save_path}/{base_path}_train.csv", index=False)
df_valid.to_csv(f"{save_path}/{base_path}_valid.csv", index=False)
df_test.to_csv(f"{save_path}/{base_path}_test.csv", index=False)

def save_results_to_csv(metrics, filename):
    fieldnames = metrics.keys()
    data_row = metrics.values()
    file_exists = os.path.exists(filename)
    
    try:
        with open(filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists or os.stat(filename).st_size == 0:
                writer.writerow(fieldnames)
                print(f"Created new CSV file: '{filename}' and wrote header.")
            writer.writerow(data_row)
            print(f"Appended new results to '{filename}'.")

    except Exception as e:
        print(f"An error occurred while writing to CSV: {e}")



RESULTS_FILE = "/Users/mihajlobulesnij/Documents/system/RAI/project/DRL_DGM/data_generated_pretrained/data_generation_info.csv"
metrics["costraint_path"] = constraint_path
metrics["seed"] = seed

save_results_to_csv(metrics, RESULTS_FILE)
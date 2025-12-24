import torch
import pandas as pd
import torch.nn as nn
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch.nn.functional as F

from argparse_generator import parse_arguments
from utils_generator import generate_cnf, RandomGenerator, fix_vector_with_maxsat, cnf_to_smt_over_reals, save_results_to_csv
from DRL.constraints_code.correct_predictions import correct_preds, check_all_constraints_sat
from DRL.constraints_code.parser import parse_constraints_file
from DRL.constraints_code.compute_sets_of_constraints import compute_sets_of_constraints

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
    
# Fix parametrs
params = parse_arguments()
seed_everything(seed=params["seed"])

exp_path = os.path.join(params["save_path"], f"{params['c']}_{params['r']}_{params['b']}_{params['seed']}")

os.makedirs(exp_path, exist_ok=True)

# generate dataset
X = torch.rand(params["n_samples"],params["n_inputs"], dtype=torch.float) * 10

bounds = np.linspace(0, 10, params["c"] * params["b"] + 1)

# Split input space on B * C regions
X_regions = []
x0 = X[:, 0]
for i in range(params["c"] * params["b"]):
    left, right = bounds[i], bounds[i+1]
    if i < params["c"]*params["b"] - 1:
        mask = (x0 >= left) & (x0 < right)
    else:
        mask = (x0 >= left) & (x0 <= right)

    X_regions.append(X[mask])

# Generate C constraints 
file_paths = []
for i in range(params["c"]):
    file_path = generate_cnf(n_vars = params["n_outputs"], n_clauses = int(params["n_outputs"] * params["r"]),
                                   index = i, seed = params["seed"], exp_path = exp_path)
    if params["is_real"]:
        file_path = cnf_to_smt_over_reals(file_path, f"n{params['n_outputs']}_i{params['seed']}_{i}", exp_path = exp_path)
    file_paths.append(file_path)

random_mapping = [i for j in range(params["b"]) for i in range(params["c"])]
np.random.shuffle(random_mapping)

# Generate consistent with constraints X-Y mapping
model = RandomGenerator(params["n_inputs"], params["n_outputs"], real = True)
     
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

# Pretrain model
for i in range(params["epochs"]):
    for region_index, region in enumerate(X_regions):
        optimizer.zero_grad()
        outputs = model(region)
        col_mean = outputs.mean(dim=0)  
        col_var = ((outputs - col_mean)**2).mean(dim=0)  
        diversity_loss = 1.0 / (col_var + 1e-6) 
        diversity_loss = diversity_loss.mean()
        
        if params["is_real"] == False:
            fixed = fix_vector_with_maxsat(outputs.detach()>0.5, file_paths[random_mapping[region_index]])
            fixed = torch.stack([torch.from_numpy(fx) for fx in fixed])
            loss_constr = torch.mean(torch.abs(torch.log(1e-6 + 1 - torch.abs(fixed - outputs))))
        else:
            ordering, constraints = parse_constraints_file(file_paths[random_mapping[region_index]])
            sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
            target = correct_preds(outputs.detach(), ordering, sets_of_constr)
            target = torch.tensor(target, dtype=outputs.dtype)
            loss_constr = F.l1_loss(outputs, target)
            
        loss = diversity_loss + loss_constr
        loss.backward()
        optimizer.step()

# Generate input - output mapping consistent with constraints
Y_regions = []
dataset_info = defaultdict(list)
constraints_list = []
for region_index, region in enumerate(X_regions): 
    outputs = model(region).detach()
    if params["is_real"] == False:
        fixed = fix_vector_with_maxsat(outputs.numpy()>0.5, file_paths[random_mapping[region_index]])
        dataset_info[f"n_changes"].append(np.mean(np.all((outputs > 0.5) == fixed, axis=1)))
        dataset_info[f"hamming_distances"].append(np.mean(np.sum(fixed != outputs, axis=1)))
        Y_regions.append(fixed)
    else:
        ordering, constraints = parse_constraints_file(file_paths[random_mapping[region_index]])
        sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
        fixed = correct_preds(outputs, ordering, sets_of_constr).numpy()
        dataset_info[f"L1_distance"].append(np.mean(np.sum(np.abs(fixed - outputs.numpy()), axis=1)))
        Y_regions.append(fixed)
    constraints_list.extend([file_paths[random_mapping[region_index]] for i in range(len(region))])
        

X_regions = np.concatenate(X_regions)
Y_regions = np.concatenate(Y_regions)
constraints_list = np.expand_dims(constraints_list, axis=1)
dataset_np = np.hstack([X_regions, Y_regions, constraints_list])

input_cols = [f"noise_{i}" for i in range(X_regions.shape[1])]
output_cols = [f"pred_{i}" for i in range(Y_regions.shape[1])]
columns = input_cols + output_cols + ["constraint"]

df = pd.DataFrame(dataset_np, columns=columns)

df_train, df_temp = train_test_split(df, test_size=0.3, random_state=params["seed"])
df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=params["seed"])

df_train.to_csv(f"{exp_path}/train.csv", index=False)
df_valid.to_csv(f"{exp_path}/valid.csv", index=False)
df_test.to_csv(f"{exp_path}/test.csv", index=False)

params.update(dataset_info)
params["experiment_path"] = exp_path

RESULTS_FILE = os.path.join(params["save_path"], "dataset_info.csv")
        
save_results_to_csv(params, RESULTS_FILE)
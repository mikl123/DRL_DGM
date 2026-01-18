import torch
import pandas as pd
import torch.nn as nn
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from argparse_generator import parse_arguments
from utils_generator import generate_cnf, RandomGenerator, fix_vector_with_maxsat, cnf_to_smt_over_reals, save_results_to_csv, fix_with_smt, cnf_to_smt_over_reals_input, cnf_to_smt_over_reals_new, cnf_to_smt_over_reals_regions
from DRL.constraints_code.correct_predictions import correct_preds, check_all_constraints_sat
from DRL.constraints_code.parser import parse_constraints_file
from DRL.constraints_code.compute_sets_of_constraints import compute_sets_of_constraints

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

exp_path = os.path.join(params["save_path"], f"{params['c']}_{params['r']}_{params['b']}_{params['m']}_{params['seed']}")

os.makedirs(exp_path, exist_ok=True)

# generate dataset
X = 0.1 + torch.rand(params["n_samples"], params["n_inputs"], dtype=torch.float) * 0.8

bounds = np.linspace(0.1, 0.9, params["c"] * params["b"] + 1)

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
        if params["input_constr"]:
            file_path = cnf_to_smt_over_reals_input(file_path, f"n{params['n_outputs']}_i{params['seed']}_{i}", exp_path = exp_path, margin=float(params["m"]), n_inputs=params["n_inputs"])
        else:
            # file_path = cnf_to_smt_over_reals(file_path, f"n{params['n_outputs']}_i{params['seed']}_{i}", exp_path = exp_path, margin=float(params["m"]))
            file_path = cnf_to_smt_over_reals_regions(file_path, f"n{params['n_outputs']}_i{params['seed']}_{i}", exp_path = exp_path, n_inputs=params["n_inputs"])
    file_paths.append(file_path)

random_mapping = [i for j in range(params["b"]) for i in range(params["c"])]
np.random.shuffle(random_mapping)

# Generate consistent with constraints X-Y mapping
model = RandomGenerator(params["n_inputs"], params["n_outputs"], hidden_dim=64, real = True)
     
optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

# Pretrain model

file_paths_list = [file_paths[random_mapping[index]] for index, reg in enumerate(X_regions) for _ in range(len(reg))]
dataset = MyDataset(torch.cat(X_regions), file_paths_list)

dataloader = DataLoader(
    dataset,
    batch_size=512,
    shuffle=True
)

diversity_loss_list = []
constraint_loss_list = []
for i in range(params["epochs"]):
    loss_constr_lst = []
    loss_diversity_lst = []
    for inp, constr in dataloader:
        optimizer.zero_grad()
        constr = np.array(constr)
        outputs = model(inp)
        div_loss = 0
        
        for c in set(constr):
            constr_indices = constr == c
            outputs_constr = outputs[constr_indices]
            col_mean = outputs_constr.mean(dim=0)  
            col_var = ((outputs_constr - col_mean)**2).mean(dim=0)  
            diversity_loss = 1.0 / (col_var + 1e-6) 
            diversity_loss = diversity_loss.mean()
            div_loss+=diversity_loss
        
        if params["is_real"] == False:
            # fixed = fix_vector_with_maxsat(outputs.detach()>0.5, file_paths[random_mapping[region_index]])
            # fixed = torch.stack([torch.from_numpy(fx) for fx in fixed])
            # loss_constr = torch.mean(torch.abs(torch.log(1e-6 + 1 - torch.abs(fixed - outputs))))
            pass
        else:
            fixed = fix_with_smt(outputs.detach().clone().numpy(), inputs = inp, paths=constr)
            fixed = torch.tensor(fixed, dtype=outputs.dtype)
            loss_constr = torch.nn.functional.l1_loss(outputs, fixed)
        
        loss_constr_lst.append(loss_constr.detach().numpy())
        loss_diversity_lst.append(div_loss.detach().numpy())
        if loss_constr != 0:
            coef = float(div_loss) / float(loss_constr) 
        else:
            coef = 0
        loss = div_loss + loss_constr * coef
        
        loss.backward()
        optimizer.step()
    print(f"Epoch {i}", f"Constr loss {np.mean(loss_constr_lst):4f}", f"Diversity loss {np.mean(loss_diversity_lst):4f}")
    diversity_loss_list.append(np.mean(loss_diversity_lst))
    constraint_loss_list.append(np.mean(loss_constr_lst))

# Generate input - output mapping consistent with constraints
Y_regions = []
dataset_info = defaultdict(list)
constraints_list = []
region_list = []
n_fixes = 0
for region_index, region in enumerate(X_regions): 
    outputs = model(region).detach()
    if params["is_real"] == False:
        # fixed = fix_vector_with_maxsat(outputs.numpy()>0.5, file_paths[random_mapping[region_index]])
        # dataset_info[f"n_changes"].append(np.mean(np.all((outputs > 0.5) == fixed, axis=1)))
        # dataset_info[f"hamming_distances"].append(np.mean(np.sum(fixed != outputs, axis=1)))
        # Y_regions.append(fixed)
        pass
    else:
        outputs_np = outputs.detach().numpy()
        fixed = fix_with_smt(outputs_np, inputs = region, paths=[file_paths[random_mapping[region_index]] for _ in range(len(outputs))])
        n_fixes += len(fixed) - np.sum(np.all(np.isclose(fixed, outputs_np, atol=0.0001), axis=1))
        Y_regions.append(fixed)
        dataset_info[f"L1_distance"].append(float(np.mean(np.sum(np.abs(fixed - outputs.numpy()), axis=1))))
        fixed = torch.tensor(fixed)
        col_mean = fixed.mean(dim=0)
        col_var = ((fixed - col_mean)**2).mean(dim=0)  
        diversity_loss = 1.0 / (col_var + 1e-6) 
        diversity_loss = diversity_loss.mean()
        dataset_info[f"diversity_loss"].append(float(diversity_loss))
    constraints_list.extend([file_paths[random_mapping[region_index]] for i in range(len(region))])
    region_list.extend([region_index for i in range(len(region))])

X_regions = np.concatenate(X_regions)
Y_regions = np.concatenate(Y_regions)
constraints_list = np.expand_dims(constraints_list, axis=1)
region_list = np.expand_dims(region_list, axis=1)
dataset_np = np.hstack([X_regions, Y_regions, constraints_list, region_list])

input_cols = [f"noise_{i}" for i in range(X_regions.shape[1])]
output_cols = [f"pred_{i}" for i in range(Y_regions.shape[1])]
columns = input_cols + output_cols + ["constraint", "region"]

df = pd.DataFrame(dataset_np, columns=columns)

df_train, df_temp = train_test_split(df, test_size=0.3, random_state=params["seed"])
df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=params["seed"])

df_train.to_csv(f"{exp_path}/train.csv", index=False)
df_valid.to_csv(f"{exp_path}/valid.csv", index=False)
df_test.to_csv(f"{exp_path}/test.csv", index=False)

params.update(dataset_info)
params["experiment_path"] = exp_path
params["constraint_loss_list"] = constraint_loss_list
params["diversity_loss_list"] = diversity_loss_list
params["n_fixes"] = n_fixes
RESULTS_FILE = os.path.join(params["save_path"], "dataset_info.csv")
        
save_results_to_csv(params, RESULTS_FILE)
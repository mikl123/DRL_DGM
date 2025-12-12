import torch
import pandas as pd
import torch.nn as nn
import subprocess
import os

import random
import numpy as np
from sklearn.model_selection import train_test_split

from pysat.formula import CNF
from pysat.solvers import Minisat22, Glucose3, Lingeling

import torch
import torch.nn as nn
import numpy as np
from pysat.formula import CNF, WCNF
from pysat.solvers import Solver
from pysat.examples.rc2 import RC2
from tqdm import tqdm

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
    
import argparse

parser = argparse.ArgumentParser(description="Your script description here")

parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility"
)

parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    help="Path to the constraint file"
)

parser.add_argument(
    "--c",
    type=str,
    default=None,
)

parser.add_argument(
    "--b",
    type=str,
    default=None,
)

parser.add_argument(
    "--r",
    type=str,
    default=None,
)

args = parser.parse_args()

seed = int(args.seed)
c = int(args.c) # (Number of constraints)
r = float(args.r) # (complexity of constraint)
b = int(args.b) # (Number of disjoint regions for specific constraints)

seed_everything(seed=seed)

N_inputs = 4
N_outputs = 10
n_samples = 5000

dir = args.save_path
exp_path = os.path.join(dir, f"{c}_{r}_{b}_{seed}")
os.makedirs(exp_path, exist_ok=True)

def generate_cnf(n_vars, n_clauses, extra_name, index):
    """
    Generate a random 3-CNF formula using cnfgen and save to file.
    """
    filename = f"n{n_vars}_{extra_name}_i{seed}.cnf"
    filepath = os.path.join(exp_path, filename)
    with open(filepath, "w") as f:
        subprocess.run(
        [
            "cnfgen",         # command
            "--seed", str(seed) + str(index),  # seed argument
            "randkcnf",       # subcommand
            "3",              # k
            str(n_vars),      # number of variables
            str(n_clauses),   # number of clauses
            "--plant"         # plant a solution
            
        ],
        stdout=f,
        check=True
    )
    return filepath, filename

class RandomBooleanGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        logits = self.net(x)
        probs = torch.sigmoid(logits)
        return probs

def is_valid_sat(vec, cnf):
    """
    Check if a boolean vector satisfies CNF constraints using a SAT solver.
    vec: np.array of 0/1 values
    cnf: pysat.formula.CNF object
    """
    with Solver(bootstrap_with=cnf.clauses) as solver:
        assumptions = []
        for i, val in enumerate(vec, start=1):
            lit = i if val == 1 else -i
            assumptions.append(lit)
        return solver.solve(assumptions=assumptions)

# --- Fix invalid vectors with MaxSAT ---
def fix_vector_with_maxsat(vec, cnf_file):
    hard_cnf = CNF(from_file=cnf_file)
    formula = WCNF()

    # Add hard clauses
    for clause in hard_cnf.clauses:
        formula.append(clause, weight=None)

    n_vars = len(vec)
    
    # Soft clauses to encourage matching original vector bits
    for i, val in enumerate(vec, start=1):
        lit = i if val == 1 else -i
        formula.append([lit], weight=1)

    with RC2(formula) as rc2:
        model = rc2.compute()

    if model is None:
        raise ValueError("No satisfying assignment found for vector")

    fixed_vec = np.zeros(n_vars, dtype=np.int8)
    for lit in model:
        var = abs(lit)
        val = lit > 0
        fixed_vec[var - 1] = int(val)

    return fixed_vec

def generate_dataset(inputs):
    model = RandomBooleanGenerator(N_inputs, N_outputs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    for i in range(5):
        optimizer.zero_grad()
        outputs = model(inputs)
        col_mean = outputs.mean(dim=0)  
        col_var = ((outputs - col_mean)**2).mean(dim=0)  
        diversity_loss = 1.0 / (col_var + 1e-6) 
        diversity_loss = diversity_loss.mean()
        diversity_loss.backward()
        optimizer.step()
        
    outputs = model(inputs).detach().numpy()>0.5

    return outputs.astype(np.int8)

def fix_dataset(outputs, constraint_path = ""):
    fixed_outputs = []

    hard_cnf = CNF(from_file=constraint_path)

    valid_count = 0
    for i, vec in enumerate(tqdm(outputs)):
        if is_valid_sat(vec, hard_cnf):
            fixed_outputs.append(vec)
            valid_count += 1
        else:
            fixed_vec = fix_vector_with_maxsat(vec, constraint_path)
            fixed_outputs.append(fixed_vec)

    fixed_outputs = np.array(fixed_outputs)
    
    return fixed_outputs, len(outputs) - valid_count, np.sum(fixed_outputs != outputs)/len(outputs)


import numpy as np

X = np.random.rand(n_samples, N_inputs) * 10
X = torch.tensor(X).to(torch.float)
Y = generate_dataset(X)

bounds = np.linspace(0, 10, c*b + 1)

x0 = X[:, 0] 

X_regions = []
Y_regions = []

for i in range(c*b):
    left, right = bounds[i], bounds[i+1]
    if i < c*b - 1:
        mask = (x0 >= left) & (x0 < right)
    else:
        mask = (x0 >= left) & (x0 <= right)

    X_regions.append(X[mask])
    Y_regions.append(Y[mask])

file_paths = []
for i in range(c):
    file_paths.append(generate_cnf(N_outputs, int(N_outputs * r), f"reg_{i}", i)[0])

random_mapping = [i for j in range(b) for i in range(c)]
np.random.shuffle(random_mapping)

invalid_count = []
constraints = []
Y_regions_fixed = []
hamming_distances = []
for index in range(c*b):
    fixed_outputs, invalid_c, hamming_dist = fix_dataset(Y_regions[index], file_paths[random_mapping[index]])
    Y_regions_fixed.append(fixed_outputs)
    hamming_distances.append(hamming_dist)
    invalid_count.append(invalid_c)
    constraints.extend([file_paths[random_mapping[index]] for i in range(len(Y_regions[index]))])


combined = []
for i in Y_regions_fixed:
    print(len(np.unique(i, axis=0)))
    combined.append(len(np.unique(i, axis=0)))

X_regions = np.concatenate(X_regions)
Y_regions_fixed = np.concatenate(Y_regions_fixed)
Y_regions = np.concatenate(Y_regions)
constraints = np.expand_dims(constraints, axis=1)
dataset_np = np.hstack([X_regions, Y_regions_fixed, constraints])

input_cols = [f"noise_{i}" for i in range(X_regions.shape[1])]
output_cols = [f"pred_{i}" for i in range(Y_regions_fixed.shape[1])]
columns = input_cols + output_cols + ["constraint"]

df = pd.DataFrame(dataset_np, columns=columns)

df_train, df_temp = train_test_split(df, test_size=0.3, random_state=seed)
df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=seed)

df_train.to_csv(f"{exp_path}/train.csv", index=False)
df_valid.to_csv(f"{exp_path}/valid.csv", index=False)
df_test.to_csv(f"{exp_path}/test.csv", index=False)


data_info = [
    ("c", c),
    ("b", b),
    ("r", r),
    ("fixes", tuple(invalid_count)),
    ("combined", tuple(combined)),
    ("total_combined", len(np.unique(Y_regions_fixed, axis=0))),
    ("total_combined_all", len(np.unique(Y_regions, axis=0))),
    ("random_mapping", tuple(random_mapping)),
    ("bounds", tuple(bounds)),
    ("hamming_distances", tuple(hamming_distances))
]

RESULTS_FILE = os.path.join(dir, "dataset_info.csv")
import csv
def save_results_to_csv(metrics, filename):
    fieldnames = [name for name, value in metrics]
    data_row = [value for name, value in metrics]
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
        
        
save_results_to_csv(data_info, RESULTS_FILE)
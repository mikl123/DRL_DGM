import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from model1 import train_model1, inference_model1
from model2 import train_model2, inference_model2
from DRL.constraints_code.parser import parse_constraints_file
from DRL.constraints_code.correct_predictions import check_all_constraints_sat
# --------------------------
# Sequential Pipeline
# --------------------------
import random
import numpy as np
import torch
from torch.distributions import Normal
from sklearn.metrics import mean_absolute_error
import pandas as pd
from z3 import *
import csv
torch.set_printoptions(precision=10)
from variables import TOLERANCE
RESULTS_FILE = "model_results/model_evaluation_results.csv"

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


def extract_inequalities(expr):
    if expr.num_args() == 0:
        return []

    op = expr.decl().kind()

    if op in [Z3_OP_LT, Z3_OP_LE, Z3_OP_GT, Z3_OP_GE]:
        return [expr]

    # Logical groups → recurse
    if op in [Z3_OP_OR, Z3_OP_AND]:
        inequalities = []
        for child in expr.children():
            inequalities.extend(extract_inequalities(child))
        return inequalities

    return []

def check_vectors_against_smt2_z3(smt2_path, vectors, tolerance = 0):
    assertions = parse_smt2_file(smt2_path[:-4] + ".smt2")
    var_names = [f"x{i+1}" for i in range(len(vectors[0]))]
    z3_vars = {name: Real(name) for name in var_names}
    results = []
    for vec in vectors:
        if len(vec) != len(var_names):
            raise ValueError(f"Vector length {len(vec)} does not match number of vars {len(var_names)}")
        subs_dict = {z3_vars[name]:RealVal(float(val)) for name, val in zip(var_names, vec)}
        all_true = True
        for index, a in enumerate(assertions):
            subs_buf = []
            inequalities = [str(i) for i in extract_inequalities(a)]
            for i in inequalities:
                var_n, op, _ = i.split(" ")
                subs_buf.append((z3_vars[var_n], subs_dict[z3_vars[var_n]] + tolerance if op == ">=" else subs_dict[z3_vars[var_n]] - tolerance))
            simplified = simplify(substitute(a, *subs_buf))
            if simplified == False:
                all_true = False
                break

        results.append(all_true)
    return 1 - (sum(results)/len(results))


def check_vectors_against_smt2(smt2_path, vectors, tolerance = 0):
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

def load_data():
    constrain_name = constraint_path.split("/")[-1][:-4]
    data_folder = constraint_path.split("/")[-2]
    if "random" in data_folder:
        data_folder = "data_smt_random"
    else:
        data_folder = "data_smt"
    
    train_df = pd.read_csv(f"data_generated/{data_folder}/{constrain_name}_{dataset_index}_train.csv")
    valid_df = pd.read_csv(f"data_generated/{data_folder}/{constrain_name}_{dataset_index}_valid.csv")
    test_df = pd.read_csv(f"data_generated/{data_folder}/{constrain_name}_{dataset_index}_test.csv")
    
    input_cols = [index for index, col in enumerate(train_df.columns) if col.startswith("noise")]
    output_cols = [index for index, col in enumerate(train_df.columns) if col.startswith("pred")]

    # Train split
    X_train = train_df.iloc[:,input_cols].values
    y_train = train_df.iloc[:,output_cols].values

    # Validation split
    X_valid = valid_df.iloc[:,input_cols].values
    y_valid = valid_df.iloc[:,output_cols].values

    # Test split
    X_test = test_df.iloc[:,input_cols].values
    y_test = test_df.iloc[:,output_cols].values

    return (
    torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), 
    torch.tensor(X_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32),
    torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))


def evaluate(prediction, y_test):
    mu = prediction["mu"]
    sigma = prediction["sigma"]
    
    y_true = y_test.detach().cpu().numpy()
    mu_np = mu.detach().cpu().numpy()

    mae = mean_absolute_error(y_true, mu_np)
    
    dist = Normal(mu, sigma)
    likelihood = dist.log_prob(y_test).mean().item()

    print(f"MAE: {mae:.4f}")
    print(f"Average log-likelihood: {likelihood:.4f}")

    return mae, likelihood


def train_pipeline(x_train, y_train, x_val, y_val, x_test, y_test, config):
    model1 = train_model1(x_train, y_train, x_val, y_val, config=config["model_1"])
    model_1_predicted = inference_model1(model1, x_test, config=config["model_1"])
    
    model2 = train_model2(model1, x_train, y_train, x_val, y_val, config=config)
    model_2_predicted = inference_model2(model1, model2, x_test, config=config)
    
    print("Model1")
    model_1_mae, model1_likelihood = evaluate(model_1_predicted, y_test)
    
    print("Model2")
    model_2_mae, model2_likelihood = evaluate(model_2_predicted, y_test)
    
    y_test_violation = check_vectors_against_smt2(constraint_path, y_test, tolerance = TOLERANCE)
    y_test_violation_z3 = check_vectors_against_smt2_z3(constraint_path, y_test, tolerance = TOLERANCE)
    
    model_1_violation = check_vectors_against_smt2(constraint_path, model_1_predicted["mu"], tolerance= TOLERANCE)
    model_1_violation_z3 = check_vectors_against_smt2_z3(constraint_path, model_1_predicted["mu"], tolerance= TOLERANCE)
    
    model_2_violation = check_vectors_against_smt2(constraint_path, model_2_predicted["mu"], tolerance= TOLERANCE)
    model_2_violation_z3 = check_vectors_against_smt2_z3(constraint_path, model_2_predicted["mu"], tolerance= TOLERANCE)
    
    arr = []
    tolerances = [TOLERANCE, 0.0001, 0.001, 0.01, 0.1]
    
    for tolerance in tolerances:
        arr.append((f"y_test_violation_{tolerance}",check_vectors_against_smt2(constraint_path, y_test, tolerance)))
    for tolerance in tolerances:
        arr.append((f"model_1_violation_{tolerance}",check_vectors_against_smt2(constraint_path, model_1_predicted["mu"], tolerance)))
    for tolerance in tolerances:
        arr.append((f"model_2_violation_{tolerance}",check_vectors_against_smt2(constraint_path, model_2_predicted["mu"], tolerance)))
    
    assert y_test_violation == y_test_violation_z3
    assert y_test_violation == 0
    assert model_1_violation == model_1_violation_z3
    assert model_2_violation == model_2_violation_z3
    
    
    all_metrics_to_save = [
        ('model_1_mae', model_1_mae),
        ('model_1_likelihood', model1_likelihood),
        ('model_2_mae', model_2_mae),
        ('model_2_likelihood', model2_likelihood),
        ('y_test_violation', y_test_violation),
        ('model_1_violation', model_1_violation),
        ('model_2_violation', model_2_violation),
        ('constraint_path', constraint_path),
        ('dataset_index',dataset_index),
        *arr
    ]
    save_results_to_csv(all_metrics_to_save, filename= RESULTS_FILE)

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
    "--dataset_index",
    type=str,
    default=None,
    help="dataset index"
)
args = parser.parse_args()

seed = args.seed
constraint_path = args.constraint_path
dataset_index = args.dataset_index

if __name__ == "__main__":
    
    config = {"model_1":{
            'batch_size': 32,
            'epochs': 10,
            'lr': 0.001,
            'hidden_dim': 100,
            'device': 'cpu',
            "patience":20
        },
            "model_2": {
            'batch_size': 32,
            'epochs': 10,
            'lr': 0.001,
            'hidden_dim': 100,
            'device': 'cpu',
            "patience":20
        }}
    
    seed_everything(seed)

    n_inputs = 20
    m_outputs = 10
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()

    train_pipeline(x_train = X_train, y_train = y_train,
                   x_val = X_valid, y_val = y_valid,
                   x_test = X_test, y_test = y_test, config = config)


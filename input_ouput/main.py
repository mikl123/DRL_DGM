import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from model1 import train_model1, inference_model1
from model2 import train_model2, inference_model2
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
from pysat.formula import CNF
from pysat.solvers import Solver

RESULTS_FILE = "model_results/input_output2_constr_fuzz.csv"

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

def load_data(experiment_path):
    
    c, r, b, _ = experiment_path.split("/")[-1].split("_")
    c = int(c)
    r = float(r)
    b = int(b)
    
    train_df = pd.read_csv(f"{experiment_path}/train.csv")
    valid_df = pd.read_csv(f"{experiment_path}/valid.csv")
    test_df = pd.read_csv(f"{experiment_path}/test.csv")
    
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

    train_constr = train_df.loc[:,"constraint"].values
    test_constr = test_df.loc[:,"constraint"].values
    valid_constr = valid_df.loc[:,"constraint"].values
    
    return (
    torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), train_constr,
    torch.tensor(X_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32), valid_constr,
    torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32), test_constr)

def is_valid_sat(vec, cnf_path):
    """
    Check if a boolean vector satisfies CNF constraints using a SAT solver.
    vec: np.array of 0/1 values
    cnf: pysat.formula.CNF object
    """
    cnf = CNF(from_file=cnf_path)
    with Solver(bootstrap_with=cnf.clauses) as solver:
        assumptions = []
        for i, val in enumerate(vec, start=1):
            lit = i if val == 1 else -i
            assumptions.append(lit)
        return solver.solve(assumptions=assumptions)

def evaluate(probs, y_test):
    mean_sigmoid_loss = F.binary_cross_entropy(probs, y_test)
    pred_binary = (probs >= 0.5).float()
    hamming_distance = (pred_binary != y_test).float().mean()
    
    correct_vectors = (pred_binary == y_test).all(dim=1).float()
    vector_accuracy = correct_vectors.mean()
    
    return mean_sigmoid_loss.item(), hamming_distance.item(), vector_accuracy

def check_vectors_against_smt(vectors, test_constr):
    counter = 0
    for index, vec in enumerate(vectors):
        if is_valid_sat(vec, cnf_path = test_constr[index]):
            counter+=1
    return (np.abs(len(vectors) - counter))/len(vectors)
            

def train_pipeline(x_train, y_train, train_constr, x_val, y_val, val_constr, x_test, y_test, test_constr, config):
    model1 = train_model1(x_train, y_train, x_val, y_val, train_constr = train_constr, val_constr = val_constr, config=config["model_1"])
    model_1_predicted = inference_model1(model1, x_test, config=config["model_1"])
    
    model2 = train_model2(model1, x_train, y_train, x_val, y_val, config=config)
    model_2_predicted = inference_model2(model1, model2, x_test, config=config)
    
    print("Model1")
    model1_loss, model1_hamming, model1_accuracy = evaluate(model_1_predicted, y_test)
    
    print("Model2")
    model2_loss, model2_hamming, model2_accuracy = evaluate(model_2_predicted, y_test)
    
    y_test_violation = check_vectors_against_smt(y_test, test_constr)
    assert y_test_violation == 0
    model_1_violation = check_vectors_against_smt(model_1_predicted>0.5, test_constr)
    model_2_violation = check_vectors_against_smt(model_2_predicted>0.5, test_constr)
    
    
    all_metrics_to_save = [
        ("experiment_path", experiment_path.split("/")[-1]),
        ("seed", seed),
        ("constraints_weight", config["model_1"]["constraints_weight"]),
        ('model1_loss', float(model1_loss)),
        ('model1_hamming', float(model1_hamming)),
        ('model1_accuracy', float(model1_accuracy)),
        ('model2_loss', float(model2_loss)),
        ('model2_hamming', float(model2_hamming)),
        ('model2_accuracy', float(model2_accuracy)),
        ('y_test_violation', float(y_test_violation)),
        ('model_1_violation', float(model_1_violation)),
        ('model_2_violation', float(model_2_violation)),
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
    "--experiment_path",
    type=str,
    default=None,
    help="Path to the experiment folder"
)


parser.add_argument(
    "--constraints_weight",
    type=float,
    default=None,
    help="Constraint weight"
)



args = parser.parse_args()

seed = int(args.seed)
experiment_path = args.experiment_path
constraints_weight = float(args.constraints_weight)

if __name__ == "__main__":
    
    config = {"model_1":{
            'batch_size': 32,
            'epochs': 100,
            'lr': 0.001,
            'hidden_dim': 100,
            'device': 'cpu',
            "patience":20,
            "constraints_weight": constraints_weight
        },
            "model_2": {
            'batch_size': 32,
            'epochs': 100,
            'lr': 0.001,
            'hidden_dim': 100,
            'device': 'cpu',
            "patience":20
        }}
    
    seed_everything(seed)

    X_train, y_train, train_constr, X_valid, y_valid, val_constr, X_test, y_test, test_constr = load_data(experiment_path)

    train_pipeline(x_train = X_train, y_train = y_train, train_constr = train_constr,
                   x_val = X_valid, y_val = y_valid, val_constr = val_constr,
                   x_test = X_test, y_test = y_test, test_constr = test_constr, config = config)


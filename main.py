import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import random
import numpy as np
import torch
from z3 import *

torch.set_printoptions(precision=10)


from model1 import train_model1, inference_model1
from model2 import train_model2, inference_model2
from utils_model import evaluate_prep, check_vectors_against_sat, load_data, evaluate_real, check_vectors_against_smt2, calculate_constr_loss_real
from utils_generator import save_results_to_csv

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

def train_pipeline(x_train, y_train, train_constr, y_train_unsup, x_val, y_val, val_constr, y_val_unsup, x_test, y_test, test_constr, test_region, config):
    
    model1 = train_model1(x_train, y_train, y_train_unsup, x_val, y_val, y_val_unsup,train_constr = train_constr, val_constr = val_constr, config=config["model_1"], is_real = is_real)
    model_1_predicted = inference_model1(model1, x_test, config=config["model_1"])
    
    model2 = train_model2(model1, x_train, y_train, x_val, y_val, config=config, is_real = is_real)
    model_2_predicted = inference_model2(model1, model2, x_test, config=config)
    
    all_metrics_to_save = {}
    if is_real == False:
        model_1_predicted = torch.sigmoid(model_1_predicted)
        model_2_predicted = torch.sigmoid(model_2_predicted)
        (all_metrics_to_save["model1_loss"],
        all_metrics_to_save["model1_hamming"],
        all_metrics_to_save["model1_accuracy"]) = evaluate_prep(model_1_predicted, y_test)
        
        all_metrics_to_save["results_per_region"] = evaluate_prep(model_1_predicted, y_test, test_region = test_region)

        (all_metrics_to_save["model2_loss"],
         all_metrics_to_save["model2_hamming"],
         all_metrics_to_save["model2_accuracy"]) = evaluate_prep(model_2_predicted, y_test)
        
        
        all_metrics_to_save["y_test_violation"] = check_vectors_against_sat(y_test, test_constr)
        assert all_metrics_to_save["y_test_violation"] == 0, "True target should not violate constraints"
        all_metrics_to_save["model_1_violation"] = check_vectors_against_sat(model_1_predicted>0.5, test_constr)
        all_metrics_to_save["model_2_violation"] = check_vectors_against_sat(model_2_predicted>0.5, test_constr)
    else:
        all_metrics_to_save["results_per_region_model1"] = evaluate_real(model_1_predicted, y_test, test_region = test_region)
        all_metrics_to_save["model1_mae"] = evaluate_real(model_1_predicted, y_test)
        all_metrics_to_save["model2_mae"] = evaluate_real(model_2_predicted, y_test)
        
        # all_metrics_to_save["y_test_violation"] = check_vectors_against_smt2(y_test, test_constr, tolerance=0.00001)
        # for i in [0.0001, 0.001, 0.01]:
        #     all_metrics_to_save[f"model_1_violation_{i}"] = check_vectors_against_smt2(model_1_predicted, test_constr, tolerance=i)
        #     all_metrics_to_save[f"model_2_violation_{i}"] = check_vectors_against_smt2(model_2_predicted, test_constr, tolerance=i)
        
        all_metrics_to_save["test"] = calculate_constr_loss_real(y_test, x_test, test_constr).detach().numpy()
        assert all_metrics_to_save["test"] == 0, "True target should not violate constraints"
        all_metrics_to_save["model_1_violation_loss"] = calculate_constr_loss_real(model_1_predicted, x_test, test_constr).detach().numpy()
        all_metrics_to_save["model_2_violation_loss"] = calculate_constr_loss_real(model_2_predicted, x_test, test_constr).detach().numpy()
        
        
    all_metrics_to_save["experiment_path"] = experiment_path.split("/")[-1]
    all_metrics_to_save["seed"] = seed
    all_metrics_to_save["constraints_weight"] = config["model_1"]["constraints_weight"]
    for key in config["model_1"]:
        all_metrics_to_save["model1"+key] = config["model_1"][key]
    for key in config["model_2"]:
        all_metrics_to_save["model2"+key] = config["model_2"][key]
        
    save_results_to_csv(all_metrics_to_save, filename = RESULTS_FILE)

import argparse

parser = argparse.ArgumentParser(description="Your script description here")

parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="Random seed for reproducibility"
)

parser.add_argument(
    "--experiment_path",
    type=str,
    default="gen_dataset_new/unsup/5_5.0_1_0.1_1",
    help="Path to the experiment folder"
)

parser.add_argument(
    "--constraints_weight",
    type=float,
    default=0.5,
    help="Constraint weight"
)

parser.add_argument(
    "--results_file",
    type=str,
    default="model_results_new/test.csv",
    help="Constraint weight"
)
parser.add_argument(
    "--is_real", 
    default=True,
    action="store_true"
)

args = parser.parse_args()

seed = int(args.seed)
experiment_path = str(args.experiment_path)
constraints_weight = float(args.constraints_weight)
RESULTS_FILE = str(args.results_file)
is_real = bool(args.is_real)

if __name__ == "__main__":
    
    config = {
        "model_1":{
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
            'epochs': 0,
            'lr': 0.001,
            'hidden_dim': 100,
            'device': 'cpu',
            "patience":20
        }}
    seed_everything(seed)

    X_train, y_train, train_constr, train_region, X_valid, y_valid, val_constr, val_region, X_test, y_test, test_constr, test_region = load_data(experiment_path)
    
    indices_tr = list(range(len(y_train)))
    indices_val = list(range(len(y_valid)))
    indices_tr = random.sample(indices_tr, int(len(indices_tr) * 0.98))
    indices_val = random.sample(indices_val, int(len(indices_val) * 0.98))
    
    y_train_unsup = torch.zeros(len(y_train))
    y_train_unsup[indices_tr] = -1 
    y_val_unsup = torch.zeros(len(y_valid))
    y_val_unsup[indices_val] = -1 

    train_pipeline(x_train = X_train, y_train = y_train, train_constr = train_constr, y_train_unsup = y_train_unsup,
                   x_val = X_valid, y_val = y_valid, val_constr = val_constr, y_val_unsup = y_val_unsup,
                   x_test = X_test, y_test = y_test, test_constr = test_constr, test_region = test_region, config = config)


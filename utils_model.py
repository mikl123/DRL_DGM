import torch
import torch.nn.functional as F
# --------------------------
# Sequential Pipeline
# --------------------------
import random
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
import pandas as pd
from z3 import *
import csv
torch.set_printoptions(precision=10)
from pysat.formula import CNF
from pysat.solvers import Solver

from DRL.constraints_code.parser import parse_constraints_file
from DRL.constraints_code.correct_predictions import check_all_constraints_sat
from utils_generator import fix_vector_with_maxsat
from DRL.constraints_code.correct_predictions import correct_preds
from DRL.constraints_code.compute_sets_of_constraints import compute_sets_of_constraints

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

def evaluate_prep(probs, y_test):
    mean_sigmoid_loss = F.binary_cross_entropy(probs, y_test)
    pred_binary = (probs >= 0.5).float()
    hamming_distance = (pred_binary != y_test).float().mean()
    
    correct_vectors = (pred_binary == y_test).all(dim=1).float()
    vector_accuracy = correct_vectors.mean()
    
    return mean_sigmoid_loss.item(), hamming_distance.item(), vector_accuracy

def check_vectors_against_sat(vectors, test_constr):
    counter = 0
    for index, vec in enumerate(vectors):
        if is_valid_sat(vec, cnf_path = test_constr[index]):
            counter+=1
    return (np.abs(len(vectors) - counter))/len(vectors)
            
def check_vectors_against_smt2(vectors, constraint_paths, tolerance = 0):
    """
    Check a list of vectors against SMT2 constraints and return
    the fraction of vectors that violate at least one constraint.

    Args:
        smt2_path (str): path to the SMT2 file.
        vectors (list[dict]): list of dictionaries mapping variable names to values.

    Returns:
        float: fraction of violating vectors (0.0 = all satisfy, 1.0 = all violate)
    """
    indices_dict = {}
    for i, val in enumerate(constraint_paths):
        indices_dict.setdefault(val, []).append(i)
    sat_buf = 0
    for path in indices_dict:
        _, constraints = parse_constraints_file(path)
        for i in indices_dict[path]:
            sat = check_all_constraints_sat(vectors[i:i+1], constraints=constraints, error_raise=False, tolerance = tolerance)
            sat_buf += sat
    return 1 - (sat_buf/len(vectors))


def evaluate_real(prediction, y_test):
    y_true = y_test.detach().cpu().numpy()

    mae = mean_absolute_error(y_true, prediction)
    return mae


def calculate_fuzzy_loss(predictions, constr_path):
    hard_cnf = CNF(from_file=constr_path)
    clause_satisfactions = [[] for _ in range(len(predictions))]
    for clause in hard_cnf.clauses:
        literals_probs = []
        for lit in clause:
            var_idx = abs(lit) - 1
            prob = predictions[:,var_idx]
            if lit < 0:
                literals_probs.append(1 - prob)
            else:
                literals_probs.append(prob)
        clause_sat = torch.clamp(torch.sum(torch.stack(literals_probs), axis = 0), max=1.0)
        for i in range(len(clause_satisfactions)):
            clause_satisfactions[i].append(clause_sat[i])
    
    clause_satisfactions = [torch.stack(row) for row in clause_satisfactions]
    clause_satisfactions = torch.stack(clause_satisfactions)
    total_satisfaction = torch.mean(torch.prod(clause_satisfactions, axis = 1))

    loss = 1 - total_satisfaction
    return loss

def calculate_constr_loss_prep(predictions, constr_paths, epsilon = 0.000001, fuzzy = False):
    indices_dict = {}
    for i, val in enumerate(constr_paths):
        indices_dict.setdefault(val, []).append(i)
        
    if fuzzy == True:
        global_fuzzy = 0
        for key, value in indices_dict.items():
            loss = calculate_fuzzy_loss(predictions[value],key)
            global_fuzzy += loss*len(value)
        global_fuzzy /= len(predictions)
        return global_fuzzy
    else:
        loss_list = []
        for path in indices_dict:
            fixed = fix_vector_with_maxsat((predictions[indices_dict[path]].detach()>0.5), path)
            loss_list.append(torch.sum(torch.abs(torch.log(epsilon + 1 - torch.abs(torch.tensor(fixed) - predictions[indices_dict[path]])))))
        return torch.sum(torch.stack(loss_list))/len(predictions)
    
def calculate_constr_loss_real(predictions, constr_paths):
    indices_dict = {}
    for i, val in enumerate(constr_paths):
        indices_dict.setdefault(val, []).append(i)
    loss_list = []
    for path in indices_dict:
        ordering, constraints = parse_constraints_file(path)
        sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
        target = correct_preds(predictions[indices_dict[path]].detach().clone(), ordering, sets_of_constr)
        loss_list.append(torch.sum(torch.abs(predictions[indices_dict[path]] - target)))
    return torch.sum(torch.stack(loss_list))/len(predictions)
    

import torch
import pandas as pd
import torch.nn as nn
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from pysat.formula import CNF, WCNF
from pysat.solvers import Solver
from pysat.examples.rc2 import RC2
from tqdm import tqdm
import subprocess
import csv

from z3 import Real, Solver, Or, And, sat, Optimize, Reals, RealVal, Sum, Abs, sat
from pysat.formula import CNF

def generate_cnf(n_vars, n_clauses, index, seed = 0, exp_path = None):
    """
    Generate a random 3-CNF formula using cnfgen and save to file.
    """
    filename = f"n{n_vars}_i{seed}_{index}.cnf"
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
    return filepath

class RandomGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, real = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.real = real
        
    def forward(self, x):
        logits = self.net(x)
        probs = torch.sigmoid(logits)
        if self.real is False:
            return (probs>0.5).to(torch.int)
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
def fix_vector_with_maxsat(vectors, cnf_file):
    hard_cnf = CNF(from_file=cnf_file)
    fixed = []
    for vec in vectors:
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
        fixed.append(fixed_vec)
    return fixed


def fix_with_smt(vectors,path = None):
    opt = Optimize()

    opt.from_file(path)
    fixed = 
    for vector in vectors:
        # 2. Declare the variables
        # Note: These must match the names defined in your .smt2 file exactly
        x, y, z = Reals('x y z')

        # 3. Define the target values
        target = {
            x: RealVal("5.0"),
            y: RealVal("10.0"),
            z: RealVal("3.0")
        }

        # 4. Define the Manhattan distance objective
        distance_expr = Sum([Abs(v - target[v]) for v in target])

        # 5. Set the optimization goal
        opt.minimize(distance_expr)

        # 6. Check and evaluate
        if opt.check() == sat:
            m = opt.model()
            print("Model solution:")
            print(f"x: {m.eval(x)}, y: {m.eval(y)}, z: {m.eval(z)}")
            print("Minimum Distance =", m.eval(distance_expr))
        else:
            print("No solution found (unsat).")


def cnf_to_smt_over_reals(cnf_path, filename, random_v = False, margin = 0.1, exp_path = None):
    """
    Converts CNF formula into an SMT formula over real numbers in [0,1]
    Each variable A is replaced with A > 0.5 (for positive literal)
    and A < 0.5 (for negative literal)
    """
    z3_vars = {}
    cnf = CNF(from_file=cnf_path)
    solver = Solver()
    random_values = {}
    for var in range(1, cnf.nv + 1):
        r = Real(f"x{var}")
        z3_vars[var] = r
        if random_v is True:
            min_value = margin * 2
            max_value = 1 - margin * 2
            value = round(random.random() * (max_value - min_value) + min_value, 3)
            random_values[var] = value
        else:
            random_values[var] = 0.5
        
    str_constraint = f"ordering {' '.join([f'y_{i}' for i in range(cnf.nv)])}"
    
    for clause in cnf.clauses:
        z3_clause = []
        str_clause = "\n"
        for lit in clause:
            var = abs(lit)
            if lit > 0:
                z3_clause.append(z3_vars[var] >= round(random_values[var] + margin/2, 3))
                str_clause += f"y_{var-1} >= {round(random_values[var] + margin/2, 3)} or "
            else:
                z3_clause.append(z3_vars[var] <= round(random_values[var] - margin/2, 3))
                str_clause += f"-y_{var-1} >= -{round(random_values[var] - margin/2, 3)} or "
        str_constraint += str_clause[:-4]
        solver.add(Or(*z3_clause))

    if filename:
        with open(os.path.join(exp_path, filename+".smt2"), "w") as f:
            f.write(solver.to_smt2())
        with open(os.path.join(exp_path, filename+".txt"), "w") as f:
            f.write(str_constraint)
    return os.path.join(exp_path, filename+".txt")





def save_results_to_csv(metrics, filename):
    fieldnames = [name for name in metrics]
    data_row = [metrics[name] for name in metrics]
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
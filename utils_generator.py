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

def fix_with_smt(vectors, inputs = None, paths=None):
    fixed_vectors = []
    
    for index, vector in enumerate(vectors):
        opt = Optimize()
        if paths[index]:
            opt.from_file(paths[index])
        targets = []
        var_names = [f"x{i}" for i in range(1,len(vectors[0]) + 1)]
        for var_name, target_value in zip(var_names, vector):
            z3_var = Reals(var_name)[0] 
            diff = Abs(z3_var - RealVal(str(target_value)))
            targets.append(diff)
        distance_expr = Sum(targets)
        
        if inputs is not None:
            inp_var_names = [f"i{i}" for i in range(1,len(inputs[0]) + 1)]
            inp_z3_vars = {v: Real(v) for v in inp_var_names}
            for var_name in inp_var_names:
                opt.add(inp_z3_vars[var_name] == RealVal(str(float(inputs[index][int(var_name[1:]) - 1]))))
        opt.minimize(distance_expr)

        # 3. Solve
        if opt.check() == sat:
            m = opt.model()
            # Build a dictionary of the fixed values
            solution = [float(m.eval(Reals(v_name)[0]).as_fraction()) for v_name in var_names]
            # print([float(m.eval(Reals(v_name)[0]).as_fraction()) for v_name in inp_var_names])
            # print(inputs[index])
            fixed_vectors.append(solution)
            
        else:
            print(f"Could not satisfy closest sat for vector: {vector}")
            raise Exception("cannot find satisfiable assignment")

    return fixed_vectors


def cnf_to_smt_over_reals_input(cnf_path, filename, random_v = False, margin = 0.1, exp_path = None, n_inputs = None):
    """
    Converts CNF formula into an SMT formula over real numbers in [0,1]
    Each variable A is replaced with A > 0.5 (for positive literal)
    and A < 0.5 (for negative literal)
    """
    z3_vars = {}
    z3_inp_vars = {}
    cnf = CNF(from_file=cnf_path)
    solver = Solver()
    random_values = {}
    for var in range(1, cnf.nv + 1):
        r = Real(f"x{var}")
        inp = Real(f"i{random.randint(1, n_inputs)}")
        z3_vars[var] = r
        z3_inp_vars[var] = inp
        if random_v is True:
            min_value = margin * 2
            max_value = 1 - margin * 2
            value = round(random.random() * (max_value - min_value) + min_value, 3)
            random_values[var] = value
        else:
            random_values[var] = 0.5
        
    
    for clause in cnf.clauses:
        z3_clause = []
        str_clause = "\n"
        for lit in clause:
            var = abs(lit)
            if lit > 0:
                z3_clause.append(z3_vars[var] >= z3_inp_vars[var] + margin/2)
            else:
                z3_clause.append(z3_vars[var] <= z3_inp_vars[var] - margin/2)
        solver.add(Or(*z3_clause))

    if filename:
        with open(os.path.join(exp_path, filename+".smt2"), "w") as f:
            f.write(solver.to_smt2())
    return os.path.join(exp_path, filename+".smt2")


import random
import tempfile
from z3 import *
from pysat.formula import CNF

def cnf_to_smt_over_reals_new(
    cnf_path,
    filename,
    exp_path=None,
    n_inputs=None,
    max_retries=10
):
    cnf = CNF(from_file=cnf_path)

    for attempt in range(max_retries):
        solver = Solver()
        z3_vars = {}
        z3_inp_vars = {}
        
        # input vars
        for i in range(1, n_inputs + 1):
            inp = Real(f"i{i}")
            z3_inp_vars[i] = inp

        # CNF vars
        for var in range(1, cnf.nv + 1):
            r = Real(f"x{var}")
            z3_vars[var] = r
            solver.add(r >= 0, r <= 1)

        clauses = []
        # clauses
        for clause in cnf.clauses:
            z3_clause = []

            for lit in clause:
                var = abs(lit)
                state = random.random()

                if state < 0.5:
                    random_n = random.random() * 0.9 + 0.05
                    if lit > 0:
                        z3_clause.append(z3_vars[var] < random_n)
                    else:
                        z3_clause.append(z3_vars[var] >= random_n)
                else:
                    random_input = random.choice(list(z3_inp_vars.values()))
                    if lit > 0:
                        z3_clause.append(z3_vars[var] < random_input)
                    else:
                        z3_clause.append(z3_vars[var] >= random_input)

            clauses.append(Or(*z3_clause))

        input_space = [And(z3_inp_vars[inp] >= 0.1, z3_inp_vars[inp] <= 0.9) for inp in z3_inp_vars]
        input_vars = [z3_inp_vars[inp] for inp in z3_inp_vars]
        output_vars = [z3_vars[inp] for inp in z3_vars]
    
        forall_formula = ForAll(
            input_vars,
            Implies(
                And(*input_space), 
                Exists(output_vars, And(*clauses))
            )
        )
        
        
        solver.add(forall_formula)
        if solver.check() == sat:
            if filename:
                solver = Solver()
                solver.add(And(*clauses))
                with open(os.path.join(exp_path, filename+".smt2"), "w") as f:
                    f.write(solver.to_smt2())
            print(f"SAT on attempt {attempt+1}")
            return os.path.join(exp_path, filename+".smt2")
            
            
        print(f"UNSAT on attempt {attempt+1}, retrying...")

    raise RuntimeError("Failed to generate satisfiable instance")

def cnf_to_smt_over_reals_regions(
    cnf_path,
    filename,
    exp_path=None,
    n_inputs=None,
    max_retries=100
):
    cnf = CNF(from_file=cnf_path)

    for attempt in range(max_retries):
        solver = Solver()
        z3_vars = {}
        z3_inp_vars = {}
        
        # input vars
        for i in range(1, n_inputs + 1):
            inp = Real(f"i{i}")
            z3_inp_vars[i] = inp

        # CNF vars
        for var in range(1, cnf.nv + 1):
            r = Real(f"x{var}")
            z3_vars[var] = r
            solver.add(r >= 0, r <= 1)

        clauses = []
        # clauses
        for clause in cnf.clauses:
            z3_clause = []

            for lit in clause:
                var = abs(lit)
                left_bound = None
                right_bound = None
                
                state = random.random() 
                
                if state>0.3:
                    left_bound = random.random() * 0.75
                else:
                    left_bound = random.choice(list(z3_inp_vars.values()))
                    
                state = random.random()
                    
                if state>0.3:
                    if type(left_bound) == float:
                        right_bound = left_bound + 0.1 + random.random() * (1 - left_bound - 0.1)
                    else:
                        right_bound = random.random()
                else:
                    candidates = [v for v in z3_inp_vars.values() if v is not left_bound]
                    right_bound = random.choice(candidates)

                        
                z3_clause.append(And(left_bound < z3_vars[var],
                     z3_vars[var] < right_bound))

            clauses.append(Or(*z3_clause))
            
        input_space = [And(z3_inp_vars[inp] >= 0.1, z3_inp_vars[inp] <= 0.9) for inp in z3_inp_vars]
        input_vars = [z3_inp_vars[inp] for inp in z3_inp_vars]
        output_vars = [z3_vars[inp] for inp in z3_vars]
    
        forall_formula = ForAll(
            input_vars,
            Implies(
                And(*input_space), 
                Exists(output_vars, And(*clauses))
            )
        )
        
        
        solver.add(forall_formula)
        if solver.check() == sat:
            if filename:
                solver = Solver()
                solver.add(And(*clauses))
                with open(os.path.join(exp_path, filename+".smt2"), "w") as f:
                    f.write(solver.to_smt2())
            print(f"SAT on attempt {attempt+1}")
            return os.path.join(exp_path, filename+".smt2")
            
            
        print(f"UNSAT on attempt {attempt+1}, retrying...")

    raise RuntimeError("Failed to generate satisfiable instance")

def cnf_to_smt_over_reals(cnf_path, filename, random_v = False, margin = 0.1, exp_path = None, input_constr = False):
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
        solver.add(z3_vars[var] >= 0, z3_vars[var] <= 1)
        
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
    return os.path.join(exp_path, filename+".smt2")

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
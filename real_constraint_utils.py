import time
from z3 import Real, Solver, Or, And, sat
from pysat.formula import CNF
import random


def cnf_to_smt_over_reals(cnf, filename, random_v = False, margin = 0.1):
    """
    Converts CNF formula into an SMT formula over real numbers in [0,1]
    Each variable A is replaced with A > 0.5 (for positive literal)
    and A < 0.5 (for negative literal)
    """
    z3_vars = {}
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
                z3_clause.append(z3_vars[var] < round(random_values[var] - margin/2, 3))
                str_clause += f"-y_{var-1} > -{round(random_values[var] - margin/2, 3)} or "
        str_constraint += str_clause[:-4]
        solver.add(Or(*z3_clause))

    if filename:
        with open(filename+".smt2", "w") as f:
            f.write(solver.to_smt2())
        with open(filename+".txt", "w") as f:
            f.write(str_constraint)
    return solver


def solve_smt_formula(cnf):
    
    solver = cnf_to_smt_over_reals(cnf)
    start = time.time()
    result = solver.check()
    end = time.time()
    elapsed = end - start
    return elapsed, result == sat

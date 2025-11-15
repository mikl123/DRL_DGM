import subprocess
import time
import os
import csv
import numpy as np
from pysat.formula import CNF
from pysat.solvers import Minisat22, Glucose3, Lingeling
from tqdm import tqdm

def solve_with_solver(solver_cls, cnf):
    """
    Solve CNF using the given solver class and return elapsed time (capped at timeout).
    """
    start = time.time()
    is_sat = None
    with solver_cls(bootstrap_with=cnf) as solver:
        is_sat = solver.solve()
    end = time.time()
    return end - start, is_sat

def generate_3_sat_constraints(n_vars, formula_dir = "/"):
    solvers = {
        "Minisat22": Minisat22,
        "Glucose3": Glucose3,
        "Lingeling": Lingeling
    }
    n_values = [n_vars]        
    ratios = list(np.arange(1, 5, 0.2)) +  list(np.arange(8, 11, 0.5))              
    repetitions = 5                           
    
    def generate_cnf(n_vars, n_clauses, index, r):
        """
        Generate a random 3-CNF formula using cnfgen and save to file.
        """
        filename = f"n{n_vars}_r{r}_i{index}.cnf"
        filepath = os.path.join(formula_dir, filename)
        with open(filepath, "w") as f:
            subprocess.run(
                ["cnfgen", "randkcnf", "3", str(n_vars), str(n_clauses)],
                stdout=f,
                check=True
            )
        return filepath, filename

    csv_file = os.path.join(formula_dir, "sat_formulas.csv")
    os.makedirs(formula_dir, exist_ok=True)

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["filename", "n_vars", "n_clauses", "ratio"]
        header += [f"{s}_time" for s in solvers]
        header += ["Sat"]
        
        writer.writerow(header)

        print("Generating formulas and benchmarking...")

        for n in tqdm(n_values, desc="Variables"):
            for r in tqdm(ratios, desc="Ratios", leave=False):
                m = int(n * r)  # Number of clauses

                for rep in range(repetitions):
                    # Generate CNF formula
                    filepath, filename = generate_cnf(n, m, rep, r)
                    cnf = CNF(from_file=filepath)

                    # Solve with all solvers
                    times = []
                    sat = None
                    for sname, scls in solvers.items():
                        t, sat = solve_with_solver(scls, cnf)
                        times.append(t)

                    # Save row
                    row = [filename, n, m, r]
                    row += [round(t, 7) for t in times]
                    row += [str(sat)]
                    writer.writerow(row)

    print(f"\n✅ Benchmark complete. Results saved to '{csv_file}', formulas saved in '{formula_dir}/'")

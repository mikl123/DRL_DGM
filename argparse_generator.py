
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Constraint generator")

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="gen_dataset_new/test",
        help="Path to the constraint file"
    )

    parser.add_argument(
        "--c",
        type=str,
        default=1,
    )

    parser.add_argument(
        "--b",
        type=str,
        default=1,
    )

    parser.add_argument(
        "--r",
        type=str,
        default=1,
    )
    
    parser.add_argument(
        "--m",
        type=str,
        default=0.1,
    )
    
    parser.add_argument(
        "--real",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--input_constr",
        action="store_true",
        default=False,
        help="Disable input constraints"
    )

    parser.add_argument(
        "--n_inputs",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--n_outputs",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--lr",
        type=int,
        default=0.1,
    )

    args = parser.parse_args()
    params = {}
    params["seed"] = int(args.seed)
    params["c"] = int(args.c) # (Number of constraints)
    params["r"] = float(args.r) # (complexity of constraint)
    params["b"] = int(args.b) # (Number of disjoint regions for specific constraints)
    params["m"] = float(args.m) # (Margin for real constraint generator)
    
    params["is_real"] = bool(args.real)
    params["n_inputs"] = int(args.n_inputs)
    params["n_outputs"] = int(args.n_outputs)
    params["n_samples"] = int(args.n_samples)
    params["epochs"] = int(args.epochs)
    params["lr"] = float(args.lr)
    params["save_path"] = args.save_path
    params["input_constr"] = bool(args.input_constr)
    print(bool(args.input_constr))
    return params

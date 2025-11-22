from DRL.constraints_code.parser import parse_constraints_file
from DRL.constraints_code.feature_orderings import set_ordering
from DRL.constraints_code.compute_sets_of_constraints import compute_sets_of_constraints
import torch
import pandas as pd
import torch.nn as nn
from DRL.constraints_code.correct_predictions import correct_preds, check_all_constraints_sat

import random
import numpy as np
from sklearn.model_selection import train_test_split

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


# Simple MLP Generator
class Generator(nn.Module):
    def __init__(self, input_dim=16, output_dim=32, hidden_dim=64):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        return self.model(z)

def sample(ordering_list, sets_of_constr, n = 10000):
    input_length = 20
    output_length = 10
    
    generator = Generator(input_dim=input_length, output_dim=output_length)
    noise = torch.rand(size=(n, input_length)).float() * 5 
    
    with torch.no_grad():
        generated_data = generator(noise)
        generated_data = generated_data.clone().detach()
    generated_data = torch.tanh(generated_data * 5)/2 + 0.5
    generated_data = correct_preds(generated_data, ordering_list, sets_of_constr)
    
    sampled_data = generated_data.detach()

    return noise, sampled_data

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
    "--save_path",
    type=str,
    default=None,
    help="Path to the constraint file"
)

args = parser.parse_args()

seed = args.seed
constraint_path = args.constraint_path
save_path = args.save_path

seed_everything(seed)

label_ordering = "predefined"


constraints_file = constraint_path
ordering, constraints = parse_constraints_file(constraints_file)
sets_of_constr = compute_sets_of_constraints(ordering, constraints, verbose=True)
noise, sampled_data = sample(ordering, sets_of_constr)
print(sampled_data[:3] - 0.5)
check_all_constraints_sat(sampled_data, constraints=constraints, error_raise=True)

noise_np = noise.numpy()
predictions_np = sampled_data.numpy()

dataset_np = np.hstack([noise_np, predictions_np])

input_cols = [f"noise_{i}" for i in range(noise_np.shape[1])]
output_cols = [f"pred_{i}" for i in range(predictions_np.shape[1])]
columns = input_cols + output_cols

df = pd.DataFrame(dataset_np, columns=columns)

df_train, df_temp = train_test_split(df, test_size=0.3, random_state=seed)

df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=seed)

base_path = constraint_path.split("/")[-1][:-4] + f"_{seed}"

df_train.to_csv(f"{save_path}/{base_path}_train.csv", index=False)
df_valid.to_csv(f"{save_path}/{base_path}_valid.csv", index=False)
df_test.to_csv(f"{save_path}/{base_path}_test.csv", index=False)

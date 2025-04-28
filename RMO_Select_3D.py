import numpy as np
import pandas as pd
import torch
import gpytorch
import os
import csv

# === Load Dataset ===
csv_path = "/home/tomgorringe/results/Robust Results/FULLRMO/blackbox_log.csv"
df = pd.read_csv(csv_path)

X = df[["Camber", "Camber_Location", "Thickness", "AOA", "Velocity"]].values
y_cl = df["Cl"].values
y_cd = df["Cd"].values

print("✅ Loaded data:", len(X), "samples")

# === GP Classes and Training ===
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, meanPrior="zero"):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = (
            gpytorch.means.ConstantMean()
            if meanPrior == "max"
            else gpytorch.means.ZeroMean()
        )
        if meanPrior == "max":
            self.mean_module.constant.data = torch.max(train_y).clone().detach()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )

def train_gp(X, y, meanPrior="zero"):
    X_tensor = torch.tensor(X, dtype=torch.float64)
    y_tensor = torch.tensor(y, dtype=torch.float64)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    model = ExactGPModel(X_tensor, y_tensor, likelihood, meanPrior).double()
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(250):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = -mll(output, y_tensor)
        loss.backward()
        optimizer.step()

    return model

def BOGPEval(model, X_new):
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = model(torch.tensor(X_new, dtype=torch.float64))
    return pred.mean.numpy(), pred.stddev.numpy()

# === Train GPs ===
gp_cl = train_gp(X, y_cl, meanPrior="max")
gp_cd = train_gp(X, y_cd, meanPrior="max")
print("✅ Trained GPs")

# === Randomly Pick a New Test Point ===
np.random.seed(42)

lower_bounds = [df["Camber"].min(), df["Camber_Location"].min(), df["Thickness"].min(), df["AOA"].min(), df["Velocity"].min()]
upper_bounds = [df["Camber"].max(), df["Camber_Location"].max(), df["Thickness"].max(), df["AOA"].max(), df["Velocity"].max()]

def sample_random_point(lower_bounds, upper_bounds):
    return np.random.uniform(low=lower_bounds, high=upper_bounds)

new_test_point = sample_random_point(lower_bounds, upper_bounds).reshape(1, -1)

print("\n🎯 Random Test Point (Camber, Camber_Loc, Thickness, AOA, Velocity):")
print(new_test_point)

# === GP Predictions ===
pred_cl, std_cl = BOGPEval(gp_cl, new_test_point)
pred_cd, std_cd = BOGPEval(gp_cd, new_test_point)

print("\n📈 GP Predictions at Sampled Point:")
print(f"Predicted Cl Mean   : {pred_cl[0]:.4f}, Std Dev: {std_cl[0]:.4f}")
print(f"Predicted Cd Mean   : {pred_cd[0]:.4f}, Std Dev: {std_cd[0]:.4f}")

# === Save to CSV ===
save_path = "/home/tomgorringe/results/Robust Results/FULLRMO/gp_test_predictions.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

write_headers = not os.path.exists(save_path)

with open(save_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if write_headers:
        writer.writerow(["Camber", "Camber_Location", "Thickness", "AOA", "Velocity", "Predicted Cl", "Predicted Cl Std", "Predicted Cd", "Predicted Cd Std"])
    writer.writerow([
        new_test_point[0,0], new_test_point[0,1], new_test_point[0,2], new_test_point[0,3], new_test_point[0,4],
        pred_cl[0], std_cl[0], pred_cd[0], std_cd[0]
    ])

print(f"\n✅ Saved sampled point and predictions to {save_path}")

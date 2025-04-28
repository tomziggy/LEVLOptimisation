import numpy as np
import pandas as pd
import torch
import gpytorch
import matplotlib.pyplot as plt

# === GP Class ===
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

# === Train GP ===
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

# === Evaluate GP ===
def BOGPEval(model, X_new):
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = model(torch.tensor(X_new, dtype=torch.float64))
    return pred.mean.numpy(), pred.stddev.numpy()

# === Load CSV ===
csv_path = "/home/tomgorringe/results/ThicknessRMO/blackbox_log.csv"



df = pd.read_csv(csv_path)

X = df[["Thickness", "AOA", "Velocity"]].values
y_cl = df["Cl"].values
y_cd = df["Cd"].values
y_cdinv = df["1/Cd"].values
y_cdcl = df["Cl/Cd"].values

print("✅ Loaded data:", len(X), "samples")

# === Train GPs ===
gp_cl = train_gp(X, y_cl, meanPrior="max")
gp_cdinv = train_gp(X, y_cd, meanPrior="max")
gp_cdcl = train_gp(X, y_cdcl, meanPrior="max")
print("✅ Trained GPs")

# === Query Settings ===
aoa_query = 19
vel_query = 10

# Query range of thickness at fixed condition
thickness_range = np.linspace(df["Thickness"].min(), df["Thickness"].max(), 200).reshape(-1, 1)
aoa_column = np.full_like(thickness_range, aoa_query)
vel_column = np.full_like(thickness_range, vel_query)
query_points = np.hstack((thickness_range, aoa_column, vel_column))

# Predict Cl and Cd
cl_mean, _ = BOGPEval(gp_cl, query_points)
cd_mean, _ = BOGPEval(gp_cdinv, query_points)
cl_cd_mean, _ = BOGPEval(gp_cdcl, query_points)

def MTCHScalarisation(targets, weights=None, a=0.1):

    cl = targets[:,0]
    cdinv = targets[:,1]
    cl_norm = (cl - np.mean(cl)) / np.std(cl)
    cdinv_norm = (cdinv - np.mean(cdinv)) / np.std(cdinv)
    targetsnormalised = np.vstack([cl_norm, cdinv_norm]).T
    f_max = np.max(targetsnormalised, axis=0)
    abs_diff = np.abs(targetsnormalised - f_max)
    if weights is None:
        weights = np.ones_like(f_max)
    weights = weights / np.sum(weights)
    weighted_diff = abs_diff * weights
    max_term = np.max(weighted_diff, axis=1)
    sum_term = np.sum(weighted_diff, axis=1)
    scalarised = -(max_term + a * sum_term)

    return scalarised

cdinv_mean = 1/cd_mean
targets = np.column_stack((cl_mean, cdinv_mean))

idx_mean = MTCHScalarisation(targets)

# Find best thickness
best_idx = np.argmax(idx_mean)
best_thickness = thickness_range[best_idx][0]
best_cl = cl_mean[best_idx]
best_cd = cdinv_mean[best_idx]
best_ratio = cl_cd_mean[best_idx]

# Output result
print(f"\n✅ Best Thickness at AOA = {aoa_query}°, Velocity = {vel_query} m/s:")
print(f"Thickness    : {best_thickness:.4f}")
print(f"Predicted Cl : {best_cl:.4f}")
print(f"Predicted 1/Cd : {best_cd:.4f}")
print(f"Predicted Cl/Cd : {best_ratio:.4f}")

# Optional Plot
plt.figure(figsize=(8, 5))
plt.plot(thickness_range, cl_mean, label="Predicted Cl")
plt.axvline(best_thickness, color='r', linestyle='--', label=f'Best: {best_thickness:.4f}')
plt.xlabel("Thickness")
plt.ylabel("Cl")
plt.title(f"Predicted Cl vs Thickness at AOA={aoa_query}, U={vel_query}")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(8, 5))
plt.plot(thickness_range, cdinv_mean, label="Predicted 1/Cd")
plt.axvline(best_thickness, color='r', linestyle='--', label=f'Best: {best_thickness:.4f}')
plt.xlabel("Thickness")
plt.ylabel("1/Cd")
plt.title(f"Predicted 1/Cd vs Thickness at AOA={aoa_query}, U={vel_query}")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.figure(figsize=(8, 5))
plt.plot(thickness_range, cl_cd_mean, label="Predicted Cl/Cd")
plt.axvline(best_thickness, color='r', linestyle='--', label=f'Best: {best_thickness:.4f}')
plt.xlabel("Thickness")
plt.ylabel("Cl/Cd")
plt.title(f"Predicted Cl/Cd vs Thickness at AOA={aoa_query}, U={vel_query}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

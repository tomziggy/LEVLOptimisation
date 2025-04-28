import numpy as np
import pandas as pd
import torch
import gpytorch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
csv_path = "/home/tomgorringe/results/Robust Results/LocationThickness/blackbox_log.csv"

df = pd.read_csv(csv_path)

X = df[["Camber_Location","Thickness", "AOA", "Velocity"]].values
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
aoa_query = 10
vel_query = 15

# Define camber location and thickness sweep ranges
camber_range = np.linspace(df["Camber_Location"].min(), df["Camber_Location"].max(), 100)
thickness_range = np.linspace(df["Thickness"].min(), df["Thickness"].max(), 100)


# Create a meshgrid of Camber_Location × Thickness
camber_grid, thickness_grid = np.meshgrid(camber_range, thickness_range)
camber_flat = camber_grid.flatten()
thickness_flat = thickness_grid.flatten()

# Repeat AOA and Velocity
aoa_flat = np.full_like(camber_flat, aoa_query)
vel_flat = np.full_like(camber_flat, vel_query)

# Final query points: shape (N, 4)
query_points = np.vstack((camber_flat, thickness_flat, aoa_flat, vel_flat)).T

# === Predict Cl, Cd, and Cl/Cd ===
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

cdinv_mean = 1 / cd_mean
targets = np.column_stack((cl_mean, cdinv_mean))
idx_mean = MTCHScalarisation(targets)

# === Best design (index into flat grid) ===
best_idx = np.argmax(idx_mean)
best_camber = camber_flat[best_idx]
best_thickness = thickness_flat[best_idx]
best_cl = cl_mean[best_idx]
best_cd = 1 / cdinv_mean[best_idx]
best_ratio = cl_cd_mean[best_idx]

print(f"\n✅ Best Design at AOA = {aoa_query}°, Velocity = {vel_query} m/s:")
print(f"Camber Loc   : {best_camber:.4f}")
print(f"Thickness    : {best_thickness:.4f}")
print(f"Predicted Cl : {best_cl:.4f}")
print(f"Predicted Cd : {best_cd:.4f}")
print(f"Predicted Cl/Cd : {best_ratio:.4f}")


# === Reshape for heatmap plotting ===
cl_map = cl_mean.reshape(len(thickness_range), len(camber_range))
cdinv_map = cdinv_mean.reshape(len(thickness_range), len(camber_range))
clcd_map = cl_cd_mean.reshape(len(thickness_range), len(camber_range))

# === Find 2D index of best point for heatmap highlighting ===
best_thickness_idx = np.abs(thickness_range - best_thickness).argmin()
best_camber_idx = np.abs(camber_range - best_camber).argmin()

# === Heatmap Plot Function ===
def plot_heatmap(data, title, cmap, cbar_label):
    fig, ax = plt.subplots(figsize=(10, 6))
    c = ax.imshow(
        data,
        extent=[camber_range.min(), camber_range.max(), thickness_range.min(), thickness_range.max()],
        origin='lower',
        aspect='auto',
        cmap=cmap
    )
    ax.scatter(best_camber, best_thickness, color='gold', edgecolor='black', s=100, marker='*', label='Best')
    ax.set_xlabel("Camber Location")
    ax.set_ylabel("Thickness")
    ax.set_title(title)
    plt.colorbar(c, label=cbar_label)
    ax.legend()
    plt.tight_layout()
    plt.show()

# === Plot All Heatmaps ===
plot_heatmap(cl_map, f"Predicted Cl at AOA={aoa_query}, U={vel_query}", cmap='viridis', cbar_label="Cl")
plot_heatmap(cdinv_map, f"Predicted 1/Cd at AOA={aoa_query}, U={vel_query}", cmap='viridis', cbar_label="1/Cd")
plot_heatmap(clcd_map, f"Predicted Cl/Cd at AOA={aoa_query}, U={vel_query}", cmap='viridis', cbar_label="Cl/Cd")

cl_scalar_map = idx_mean.reshape(len(thickness_range), len(camber_range))
plot_heatmap(cl_scalar_map, f"Predicted Scalarised Score at AOA={aoa_query}, U={vel_query}", cmap='viridis', cbar_label="Scalarised Score")


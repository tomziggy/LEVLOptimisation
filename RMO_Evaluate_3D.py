csv_path = "/home/tomgorringe/results/Robust Results/FULLRMO/blackbox_log.csv"
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

# === Scalarisation ===
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



df = pd.read_csv(csv_path)

X = df[["Camber", "Camber_Location", "Thickness", "AOA", "Velocity"]].values
y_cl = df["Cl"].values
y_cd = df["Cd"].values

print("✅ Loaded data:", len(X), "samples")

# === Train GPs ===
gp_cl = train_gp(X, y_cl, meanPrior="max")
gp_cd = train_gp(X, y_cd, meanPrior="max")

print("✅ Trained GPs")

# === Query Settings ===
aoa_query = 10
vel_query = 15
camber_loc_fixed = df["Camber_Location"].mean()
thickness_fixed = df["Thickness"].mean()

# Sweep camber values
camber_range = np.linspace(df["Camber"].min(), df["Camber"].max(), 100)
query_points_camber = np.column_stack([
    camber_range,
    np.full_like(camber_range, camber_loc_fixed),
    np.full_like(camber_range, thickness_fixed),
    np.full_like(camber_range, aoa_query),
    np.full_like(camber_range, vel_query)
])

# Predict Cl and Cd
cl_mean, _ = BOGPEval(gp_cl, query_points_camber)
cd_mean, _ = BOGPEval(gp_cd, query_points_camber)
cdinv_mean = 1 / cd_mean

# Apply MTCH scalarisation
targets = np.column_stack((cl_mean, cdinv_mean))
scalarised_scores = MTCHScalarisation(targets)

# Find best camber
best_idx_camber = np.argmax(scalarised_scores)
best_camber = camber_range[best_idx_camber]
print(f"🏆 Best Camber based on MTCH scalarisation: {best_camber:.4f}")

# === Sweep camber location and thickness at best camber
camber_loc_range = np.linspace(df["Camber_Location"].min(), df["Camber_Location"].max(), 100)
thickness_range = np.linspace(df["Thickness"].min(), df["Thickness"].max(), 100)
camber_loc_grid, thickness_grid = np.meshgrid(camber_loc_range, thickness_range)
flat_camber_loc = camber_loc_grid.flatten()
flat_thickness = thickness_grid.flatten()

query_points_2d = np.column_stack([
    np.full_like(flat_camber_loc, best_camber),
    flat_camber_loc,
    flat_thickness,
    np.full_like(flat_camber_loc, aoa_query),
    np.full_like(flat_camber_loc, vel_query)
])

# Predict Cl, Cd
cl_mean_2d, _ = BOGPEval(gp_cl, query_points_2d)
cd_mean_2d, _ = BOGPEval(gp_cd, query_points_2d)
cdinv_mean_2d = 1 / cd_mean_2d

# Scalarisation again
targets_2d = np.column_stack((cl_mean_2d, cdinv_mean_2d))
scalarised_scores_2d = MTCHScalarisation(targets_2d)

# Find best camber location and thickness
best_idx_2d = np.argmax(scalarised_scores_2d)
best_camber_loc = flat_camber_loc[best_idx_2d]
best_thickness = flat_thickness[best_idx_2d]

print("\n🏆 Best Full Design Found:")
print(f"Camber        : {best_camber:.4f}")
print(f"Camber Loc    : {best_camber_loc:.4f}")
print(f"Thickness     : {best_thickness:.4f}")

# === Heatmaps ===
def plot_heatmap(data, title, cmap, label):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        data,
        extent=[camber_loc_range.min(), camber_loc_range.max(), thickness_range.min(), thickness_range.max()],
        origin='lower',
        aspect='auto',
        cmap=cmap
    )
    ax.scatter(best_camber_loc, best_thickness, color='gold', edgecolor='black', s=100, marker='*', label='Best')
    ax.set_xlabel("Camber Location")
    ax.set_ylabel("Thickness")
    ax.set_title(title)
    plt.colorbar(im, label=label)
    ax.legend()
    plt.tight_layout()
    plt.show()

# Reshape for plotting
cl_map = cl_mean_2d.reshape(len(thickness_range), len(camber_loc_range))
cd_map = cdinv_mean_2d.reshape(len(thickness_range), len(camber_loc_range))
scalar_map = scalarised_scores_2d.reshape(len(thickness_range), len(camber_loc_range))

plot_heatmap(cl_map, f"Predicted Cl at Best Camber={best_camber:.4f}, AOA = 10 Deg, U = 15 m/s", cmap='viridis', label="Cl")
plot_heatmap(cd_map, f"Predicted 1/Cd at Best Camber={best_camber:.4f}, AOA = 10 Deg, U = 15 m/s", cmap='viridis', label="1/Cd")
plot_heatmap(scalar_map, f"Predicted Scalarised Score at Best Camber={best_camber:.4f}, AOA = 10 Deg, U = 15 m/s", cmap='viridis', label="Scalarised Score")

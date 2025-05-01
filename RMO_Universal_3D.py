import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import subprocess
import shutil
import os
import sys
import time
import re
from itertools import product
import csv

class ExactGPModel(gpytorch.models.ExactGP): 
    def __init__(self, train_x, train_y, likelihood, meanPrior): 
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood) 

        if meanPrior == "max": # If the mean prior is set to "max", then the mean function is set to a constant mean with the constant set to the maximum value of the training data outputs.
            # self.mean_module = gpytorch.means.ZeroMean()
            self.mean_module = gpytorch.means.ConstantMean() 
            # self.mean_module.constant = torch.nn.Parameter(torch.tensor(torch.max(train_y)))
            self.mean_module.constant.data = torch.max(train_y).clone().detach()
        else:  
            self.mean_module = gpytorch.means.ZeroMean()  
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) 

    def forward(self, x):
        mean_x = self.mean_module(x) 
        covar_x = self.covar_module(x)  

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)  

def GPTrain(features, targets, meanPrior):  
    tensorSamplesXY = torch.from_numpy(features).double()  
    tensorSamplesZ = torch.from_numpy(targets).double().squeeze()     

    likelihood = gpytorch.likelihoods.GaussianLikelihood().double() 
    model = ExactGPModel(tensorSamplesXY, tensorSamplesZ, likelihood, meanPrior).double()  

    likelihood.noise = 1e-4    
    likelihood.noise_covar.raw_noise.requires_grad_(False) 

    training_iter = 250 
    model.train() 
    likelihood.train() 

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05) 
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) 

    for i in range(training_iter):
        optimizer.zero_grad()  
        output = model(tensorSamplesXY) 
        loss = -mll(output, tensorSamplesZ) 
        loss.backward() 
        optimizer.step() 

    return model 

def BOGPEval(model, newFeatures):   
    model.eval()   

    with torch.no_grad(), gpytorch.settings.fast_pred_var(): 
        observed_pred = model(torch.from_numpy(newFeatures)) 

    mean_pred = observed_pred.mean.numpy() 
    stdDev = observed_pred.stddev.numpy() 

    return mean_pred, stdDev 

def expectedImprovement(feature, currentGP, bestY, epsilon): 
    yPred, yStd = BOGPEval(currentGP, feature) 

    #to maximise
    z = (yPred - bestY - epsilon) / yStd  
    ei = ((yPred - bestY - epsilon) * norm.cdf(z)) + yStd * norm.pdf(z)

    #to minimise
    #z = (bestY - yPred - epsilon) / yStd 
    #ei = ((bestY - yPred - epsilon) * norm.cdf(z)) + yStd * norm.pdf(z) 

    return ei  

def ei_wrapper(x, currentGP, bestY, epsilon): 
    x = np.array(x).reshape(1, -1) 
    return -expectedImprovement(x, currentGP, bestY, epsilon)

def optimize_ei(bounds, currentGP, bestY, epsilon):
    result = differential_evolution( # The differential evolution optimizer is used to optimize the expected improvement.
        ei_wrapper,                  
        bounds=bounds,                 
        args=(currentGP, bestY, epsilon),  
        strategy='best1bin',          # Default strategy
        maxiter=100,                  # Maximum number of iterations
        popsize=15,                   # Population size (of DE trial points)
        tol=1e-6,                     # Tolerance for convergence
        seed=None,                    # Random seed for reproducibility
    )
    return result


def blackbox(input):
    camber, camber_loc, thickness, aoa, velocity = input

    def create_airfoil(camber, camber_loc, thickness, aoa): #function to write generate_airfoil File
                template_file = "generate_airfoil_template_3d.py"
                salome_path = os.path.join(runningdirectory,"Salome/SALOME-9.13.0-native-UB24.04-SRC/generate_airfoil.py")
                try:
                    with open(template_file, 'r') as file:
                        gen_file_contents = file.read()
                    gen_file_contents = gen_file_contents.replace("$CAMBER", str(camber))
                    gen_file_contents = gen_file_contents.replace("$CLOC", str(camber_loc))
                    gen_file_contents = gen_file_contents.replace("$THICKNESS", str(thickness))
                    gen_file_contents = gen_file_contents.replace("$AOA", str(aoa))
                    with open(salome_path, 'w') as file:
                        file.write(gen_file_contents)
                    os.chmod(salome_path, 0o755)
                    print(f"Successfully created Generate Airfoil file at {salome_path}")
                except FileNotFoundError:
                        print(f"Error: Template file '{template_file}' not found.")
                        return
                except Exception as e:
                  print(f"Error writing to {salome_path}: {e}")

    def changevel(velocity):
                template_file = "U_template" 
                zero_path = os.path.join(runningdirectory,"OpenFOAM/tomgorringe-v2312/run/optimisation_airfoil/new_case/0.orig/U")
                try:
                    with open(template_file, 'r') as file:
                        gen_file_contents = file.read()
                    gen_file_contents = gen_file_contents.replace("$U", f"{velocity} 0 0")
                    with open(zero_path, 'w') as file: 
                        file.write(gen_file_contents)
                    os.chmod(zero_path, 0o755) 
                    print(f"Successfully created U file at {zero_path}")
                except FileNotFoundError:
                        print(f"Error: Template file '{template_file}' not found.")
                        return
                except Exception as e:
                  print(f"Error writing to {zero_path}: {e}")

    def updatecoeffs(velocity):
                template_file = "forceCoeffs_template" 
                f_path = os.path.join(runningdirectory,"OpenFOAM/tomgorringe-v2312/run/optimisation_airfoil/new_case/system/forceCoeffs")
                try:
                    with open(template_file, 'r') as file:
                        gen_file_contents = file.read()
                    gen_file_contents = gen_file_contents.replace("$VEL", f"{velocity}")
                    with open(f_path, 'w') as file: 
                        file.write(gen_file_contents)
                    os.chmod(f_path, 0o755) 
                    print(f"Successfully created ForceCoeffs file at {f_path}")
                except FileNotFoundError:
                        print(f"Error: Template file '{template_file}' not found.")
                        return
                except Exception as e:
                  print(f"Error writing to {f_path}: {e}")

    
    cam = float(camber)
    loc = float(camber_loc)
    thick = float(thickness)
    angle = float(aoa)
    vel = float(velocity)
    print(f"Camber = {cam} Location = {loc} Thickness = {thick} Angle = {angle} Velocity = {vel}")
    create_airfoil(cam,loc,thick,angle)
    
    salome_shell = os.path.join(runningdirectory,"Salome/SALOME-9.13.0-native-UB24.04-SRC/salome")
    script_path = os.path.join(runningdirectory,"Salome/SALOME-9.13.0-native-UB24.04-SRC/generate_airfoil.py")

    subprocess.run(["bash", "-c", f"{salome_shell} -t {script_path}"], check=True)

    original_case = os.path.join(runningdirectory,"OpenFOAM/tomgorringe-v2312/run/optimisation_airfoil/blank_case")
    src = original_case
    dst = os.path.join(runningdirectory,"OpenFOAM/tomgorringe-v2312/run/optimisation_airfoil/new_case")

    if os.path.exists(dst):
        shutil.rmtree(dst)
        print("new_case removed")
    else:
            print("No previous cases removed")

    if not os.path.isdir(dst):
        try:
            shutil.copytree(src, dst)
            print(f"Directory copied successfully from '{src}' to '{dst}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
            sys.exit(0)

    time.sleep(5)

    stl_generated_path = os.path.join(runningdirectory,"airfoil.stl")
    stl_output_path = os.path.join(runningdirectory,"OpenFOAM/tomgorringe-v2312/run/optimisation_airfoil/new_case/constant/triSurface")

    if os.path.exists(stl_generated_path):
        shutil.copy(stl_generated_path, stl_output_path)
        print(f"STL file copied to: {stl_output_path}")
        os.remove(stl_generated_path)
    else:
        print(f"STL file not found at {stl_generated_path}")

    changevel(vel)
    print("Velocity Changed")
    updatecoeffs(vel)
    print("ForceCoeffs Updated")

    try:
        subprocess.run(["bash", "run_allrun.bashrc"], check=True)
        print(f"Ran Allrun. Check log for more details.")
    except Exception as e:
        print(f"Error executing the Allrun script: {e}")    

    def read_cl_cd(file_name):
        pth = os.path.join(runningdirectory,"OpenFOAM/tomgorringe-v2312/run/optimisation_airfoil/")
        case = file_name
        file_path = os.path.join(pth,case, 'postProcessing', 'forceCoeffs1', '0', 'coefficient.dat')
        forceRegex = r"([0-9.Ee\-+]+)\s+([0-9.Ee\-+]+)\s+([0-9.Ee\-+]+)\s+([0-9.Ee\-+]+)\s+([0-9.Ee\-+]+)\s+([0-9.Ee\-+]+)"
        
        time = []
        cl = []  # Lift coefficient
        cd = []  # Drag coefficient
        
        with open(file_path, 'r') as pipefile:
            lines = pipefile.readlines()
            
            for line in lines:
                match = re.search(forceRegex, line)
                if match:
                    time.append(float(match.group(1)))  # Time or iteration
                    cl.append(float(match.group(5)))    # Lift coefficient
                    cd.append(float(match.group(2)))    # Drag coefficient
        
        clift = cl[-1]
        cdrag = cd[-1]
        return clift, cdrag

    cl, cd = read_cl_cd('new_case')
    print("Extracting Results")
    print(f"Lift Coefficient (Cl): {cl}")
    print(f"Drag Coefficient (Cd): {cd}")

    
    L_D_ratio = cl / cd if cd != 0 else float("inf")
    cdinv = 1/cd

    print(f"Camber: {cam:.3f}")   
    print(f"Location: {loc:.3f}")      
    print(f"Thickness: {thick:.3f}")  
    print(f"Angle: {angle:.3f}")  
    print(f"Velocity: {vel:.3f}") 
    print(f"Coefficienct of Lift: {cl:.3f}")
    print(f"Lift-to-Drag Ratio (L/D): {L_D_ratio:.3f}")

    current = time.time()
    print("time=",(current - start))

    csv_path = os.path.join(runningdirectory, "results", "blackbox_log.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    headers = [
        "Camber", "Camber_Location", "Thickness",
        "AOA", "Velocity",
        "Cl", "Cd", "Cl/Cd", "1/Cd", "Time"
    ]
    row = [
        cam, loc, thick,
        angle, vel,
        cl, cd, L_D_ratio, cdinv,
        current - start
    ]
    write_headers = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_headers:
            writer.writerow(headers)
        writer.writerow(row)

    return cl, cdinv




#--------------------------------------SIM START-------------------------------------------------------------------------------------------------------
start = time.time()

#--------------------------------------SET ANY PREDEFINED VARIABLES, HYPERPARAMETERS, AND VARIABLE NAMES------------------------------------------------
#setcamber = 0.06
#camberlocation = 0.4
#maximumthickness = 0.12
epsilon = 0.2 #Expected Improvement Variable - Balances Exploration and Exploitation
alpha = 0.1 #MTCH Scalarisation Deviation Punishment Term
gpmeanprior = "max"
Variable1 = "Maximum Camber (%)"
Variable2 = "Location of Maximum Camber (x/c)"
Variable3 = "Maximum Thickness (%)"


#--------------------------------------SET RUNNING DIRECTORY-------------------------------------------------------------------------------------------
runningdirectory = '/home/tomgorringe/'
destpng = os.path.join(runningdirectory,'results')
featurestitle = ["Max Camber", "Location of Camber", "Max Thickness"]
targetstitle = ["Coeff Lift", "1 / Coeff Drag"]
condtitle = ["Angle of Attack", "Free Stream Velocity"]

#--------------------------------------SET PARAMETER AND CONDITION BOUNDS-------------------------------------------------------------------------------------------
print("Setting Feature Bounds")
lowerBound = [0,0.2,0.01] #Lower Limits of Variable1 and Variable2
upperBound = [0.15,0.8,0.25] #Upper Limits of Variable1 and Variable2
initialSampleSize = 10 #Number of Initial Parameter Sample Combinations
print(Variable1,lowerBound[0],"to",upperBound[0])
print(Variable2,lowerBound[1],"to",upperBound[1])
print(Variable3,lowerBound[2],"to",upperBound[2])

print("Setting Condition Bounds")
clowerBound = [0, 5] #Lower Limits of AOA (deg) and U (m/s)
cupperBound = [20, 25] #Upper Limits of AOA (deg) and U (m/s)
cinitialSampleSize = 5 #Number of Robust Condition Sample Combinations
print("AOA",clowerBound[0],"to",cupperBound[0])
print("U",clowerBound[1],"to",cupperBound[1])

print("Initialising Feature LHS")
sampler = qmc.LatinHypercube(d=len(lowerBound)) 
sample = sampler.random(n=initialSampleSize) 
features = np.array(qmc.scale(sample, lowerBound, upperBound))

print("Features:")
print(np.vstack([np.array(featurestitle).reshape(1, -1), features.astype(str)]))

print("Initialising Condition LHS")
csampler = qmc.LatinHypercube(d=len(clowerBound))
csample = csampler.random(n=cinitialSampleSize) 
conditions = np.array(qmc.scale(csample, clowerBound, cupperBound)) 


print("Conditions:")
print(np.vstack([np.array(condtitle).reshape(1, -1), conditions.astype(str)]))

print("Building Feature Condition Arrays")

def build_condition_matrices_for_features(features, conditions):
    condition_matrices = []
    for feature in features:
        feature_with_conditions = np.array([np.append(feature, cond) for cond in conditions])
        condition_matrices.append(feature_with_conditions)
    return condition_matrices

matrices = build_condition_matrices_for_features(features, conditions)

#------------------TRAINING DATA GENERATION LOOP---------------------------------------

clmatrix = []
cdinvmatrix = []

for idx, condition_matrix in enumerate(matrices):
    ix = idx + 1
    print(f"Matrix{ix}")
    print(condition_matrix)
    design = features[idx]
    print(f"Sample Simulation {ix}/{len(features)}. Max Camber = {design[0]} Location = {design[1]} Thickness = {design[2]}")
    print("Through AOA =", conditions[:,0], "and Velocities", conditions[:,1])
    results = np.array([blackbox(row) for row in condition_matrix])
    cl_array, cdinv_array = zip(*results)
    clmatrix.append(cl_array)
    cdinvmatrix.append(cdinv_array)
    print("Current Features:")
    print(np.vstack([np.array(featurestitle).reshape(1, -1), features.astype(str)]))
    print("Current Cls:")
    print(clmatrix)
    print("Current Cdinvs:")
    print(cdinvmatrix)
    print("time=",(time.time() - start))

def MTCH_rowwise(matrix, alpha, weights=None):
    matrix = np.array(matrix)

    means = np.mean(matrix, axis=0)
    stds = np.std(matrix, axis=0)
    normalized = (matrix - means) / stds

    f_max = np.max(normalized, axis=0)
    abs_diff = np.abs(normalized - f_max)

    if weights is None:
        weights = np.ones(normalized.shape[1])
    weights = weights / np.sum(weights)

    weighted_diff = abs_diff * weights
    max_term = np.max(weighted_diff, axis=1)
    sum_term = np.sum(weighted_diff, axis=1)
    return -(max_term + alpha * sum_term)


clscalar = MTCH_rowwise(clmatrix,alpha)
cdinvscalar = MTCH_rowwise(cdinvmatrix,alpha)
targets = np.zeros((len(clscalar), 2))
targets[:,0] = clscalar
targets[:,1] = cdinvscalar


print("Final Features:")
print(np.vstack([np.array(featurestitle).reshape(1, -1), features.astype(str)]))

print("Final Conditionally Scalarised Targets:")
print(np.vstack([np.array(targetstitle).reshape(1, -1), targets.astype(str)]))





#-------------------------------------------------------SCALARISING FUNCTION----------------------------------------------

def MTCHScalarisation(targets, weights=None, a=alpha):

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

print("Final Objective Scalarised Targets:")
scalarised = MTCHScalarisation(targets,alpha)
scalarised = (scalarised - np.mean(scalarised)) / np.std(scalarised)
print(scalarised)

#------------------------------------------------------INITIAL GP TRAINING---------------------------------------------------

globalGP = GPTrain(features[:, 0:3], scalarised, meanPrior=gpmeanprior)

best_idx = np.argmax(scalarised)
best_x = features[best_idx]
best_y = scalarised[best_idx]

# Ensure bounds now include all 3 variables
bounds = list(zip(lowerBound, upperBound))  # e.g., lowerBound = [0.06, 0.2, 0.01], upperBound = [0.06, 0.8, 0.2]

# Define the grid for EI calculation (3D)
x_range = np.linspace(lowerBound[0], upperBound[0], 20)  # Max Camber
y_range = np.linspace(lowerBound[1], upperBound[1], 20)  # Camber Location
z_range = np.linspace(lowerBound[2], upperBound[2], 20)  # Thickness

grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range)
grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])

# Evaluate GP and EI
rangePredictions, _ = BOGPEval(globalGP, grid_points)
ei = expectedImprovement(grid_points, globalGP, best_y, epsilon)
ei_history = []
rangePredictions, _ = BOGPEval(globalGP, grid_points)

ei_values = expectedImprovement(grid_points, globalGP, best_y, epsilon)
ei_surface = ei_values.reshape(grid_x.shape)

print("Plots Printed")

iteration = 0

while iteration < 100 and (iteration < 10 or np.max(ei) > 1e-7):

    iteration += 1
    print(f"Iteration {iteration}")

    scalarised = MTCHScalarisation(targets,alpha)
    scalarised = (scalarised - np.mean(scalarised)) / np.std(scalarised)
    globalGP = GPTrain(features[:, 0:3], scalarised, meanPrior=gpmeanprior)

    rangePredictions, _ = BOGPEval(globalGP, grid_points)

    best_idx = np.argmax(scalarised)
    best_x = features[best_idx]
    best_y = scalarised[best_idx]
    print(f"Current Best | Index: {best_idx}, Features: {best_x}, Targets: {best_y}")

    ei = expectedImprovement(grid_points, globalGP, best_y, epsilon)
    result = optimize_ei(bounds, globalGP, best_y, epsilon)
    new_x = result.x
    print("New candidate design (from EI):", new_x)

    max_ei = -ei_wrapper(new_x, globalGP, best_y, epsilon)
    ei_history.append(max_ei)

    print(f"Max EI at new_x: {max_ei}")

    def build_feature_condition_matrix(design, conditions):
        return np.array([[design[0], design[1], design[2], cond[0], cond[1]] for cond in conditions])

    new_matrix = build_feature_condition_matrix(new_x, conditions)
    score_array = np.array([blackbox(row) for row in new_matrix])
    cl_array, cdinv_array = zip(*score_array)
    clmatrix.append(cl_array)
    cdinvmatrix.append(cdinv_array)
    clscalar = MTCH_rowwise(clmatrix,alpha)
    cdinvscalar = MTCH_rowwise(cdinvmatrix,alpha)
    targets = np.zeros((len(clscalar), 2))
    targets[:,0] = clscalar
    targets[:,1] = cdinvscalar

    x_add = np.array([[new_x[0], new_x[1], new_x[2]]])
    features = np.vstack([features, x_add])

    scalarised = MTCHScalarisation(targets,alpha)
    scalarised = (scalarised - np.mean(scalarised)) / np.std(scalarised)

    print("Features:")
    print(np.vstack([np.array(featurestitle).reshape(1, -1), features.astype(str)]))

    print("All Cls")
    print(clmatrix)
    print("All Cds")
    print(cdinvmatrix)

    print("Conditionally Scalarised:")
    print(np.vstack([np.array(targetstitle).reshape(1, -1), targets.astype(str)]))

    print("Objectively Scalarised:")
    print(scalarised)

    ei_values = expectedImprovement(grid_points, globalGP, best_y, epsilon)
    ei_surface = ei_values.reshape(grid_x.shape)

    train_x = torch.from_numpy(features).double()
    train_y = torch.from_numpy(scalarised).double().squeeze()

    torch.save({
        'model_state_dict': globalGP.state_dict(),
        'likelihood_state_dict': globalGP.likelihood.state_dict(),
        'train_x': train_x,
        'train_y': train_y,
    }, 'results/gp_model.pth')

    print("Torch Saved")

    # Extract current targets
    cl_values = targets[:, 0]
    ld_values = targets[:, 1]
    designs = features

    # Identify Pareto front
    def pareto_front(cl, ld):
        is_pareto = np.ones(len(cl), dtype=bool)
        for i in range(len(cl)):
            for j in range(len(cl)):
                if i != j:
                    if (cl[j] >= cl[i] and ld[j] >= ld[i]) and (cl[j] > cl[i] or ld[j] > ld[i]):
                        is_pareto[i] = False
                        break
        return np.where(is_pareto)[0]

    pareto_idx = pareto_front(cl_values, ld_values)

    # Scalarisation (if needed)
    scalarised = MTCHScalarisation(targets, alpha)
    optimal_idx = np.argmax(scalarised)

    # Label each point
    point_labels = []
    for i in range(len(cl_values)):
        if i == optimal_idx:
            point_labels.append("Optimal")
        elif i in pareto_idx:
            point_labels.append("Pareto")
        else:
            point_labels.append("Dominated")

    # Plot Pareto front
    colors = {'Optimal': 'gold', 'Pareto': 'red', 'Dominated': 'blue'}
    plt.figure(figsize=(10, 7))
    for label in ['Dominated', 'Pareto', 'Optimal']:
        mask = np.array(point_labels) == label
        plt.scatter(cl_values[mask], ld_values[mask], c=colors[label],
                    label=label, s=80 if label == "Optimal" else 40, edgecolor='black')

    plt.xlabel("Mean Cl")
    plt.ylabel("Mean 1/CD")
    plt.title(f"Pareto Front - Iteration {iteration}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/pareto_front_iter{iteration}.png", dpi=300)
    plt.close()

    # Save updated Pareto CSV for this iteration
    csv_path = f"results/pareto_results.csv"
    header = ["Camber", "Location", "Thickness", "Mean_Cl", "Mean_1/Cd", "Label"]
    rows = [
        [designs[i, 0], designs[i, 1], designs[i, 2], cl_values[i], ld_values[i], point_labels[i]]
        for i in range(len(cl_values))
    ]

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Iteration {iteration}: Pareto front and CSV saved.")

    csv_path = "results/optimisation_log.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Only write header once, if file doesn't exist
    write_headers = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_headers:
            writer.writerow(["Iteration", "Camber", "Location", "Thickness", "Cl", "1/CD", "Scalarised", "EI"])
        
        row = [
            iteration,
            features[-1, 0],
            features[-1, 1],
            features[-1, 2],
            targets[-1, 0],
            targets[-1, 1],
            scalarised[-1],
            ei_history[-1] if len(ei_history) >= iteration else None
        ]
        writer.writerow(row)

    print(f"Iteration {iteration} logged to optimisation_log.csv")

print("\nOptimisation Complete!")

# Find best sampled design
best_idx = np.argmax(scalarised)
best_design = features[best_idx]
best_score = targets[best_idx]

print("Most Robust Sampled Design Found:")
print(f"Camber      : {best_design[0]:.4f}")
print(f"Location    : {best_design[1]:.4f}")
print(f"Thickness   : {best_design[2]:.4f}")
print(f"Mean Cl     : {best_score[0]:.4f}")
print(f"Mean 1/Cd   : {best_score[1]:.4f}")

# Best GP-predicted design
print("Performing Bayesian search for best predicted scalarised objective over thickness grid...")
final_preds, _ = BOGPEval(globalGP, grid_points)
best_idx_gp = np.argmax(final_preds)
best_design_gp = grid_points[best_idx_gp][0]
best_score_gp = final_preds[best_idx_gp]

print("Final GP-Based Best Design (Predicted):")
print(f"Thickness                   : {best_design_gp:.4f}")
print(f"Predicted Scalarised Score : {best_score_gp:.4f}")
print("Converged!\n")





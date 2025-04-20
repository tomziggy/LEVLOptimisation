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


class ExactGPModel(gpytorch.models.ExactGP): # ExactGP  is a PyTorch module This class defines a Gaussian Process (GP) model for exact Gaussian Process inference.
    def __init__(self, train_x, train_y, likelihood, meanPrior): # train_x: The training data inputs. train_y: The training data outputs. likelihood: The likelihood for the model. meanPrior: The prior to use for the mean function of the GP.
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood) # Initialize the parent class

        if meanPrior == "max": # If the mean prior is set to "max", then the mean function is set to a constant mean with the constant set to the maximum value of the training data outputs.
            # self.mean_module = gpytorch.means.ZeroMean()
            self.mean_module = gpytorch.means.ConstantMean() # The mean function of the GP is set to a constant mean.
            # self.mean_module.constant = torch.nn.Parameter(torch.tensor(torch.max(train_y)))
            self.mean_module.constant.data = torch.max(train_y).clone().detach() # The constant of the mean function is set to the maximum value of the training data outputs.

        else:   # If the mean prior is not set to "max", then the mean function is set to a zero mean.
            # self.mean_module = gpytorch.means.ConstantMean(constant_prior=torch.max(train_y))
            self.mean_module = gpytorch.means.ZeroMean()    # The mean function of the GP is set to a zero mean.
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())  # The covariance function of the GP is set to a Radial Basis Function (RBF) kernel.

    def forward(self, x): # x: The input data for which the GP model is evaluated.
        mean_x = self.mean_module(x) # The mean of the GP model is evaluated at the input data.
        covar_x = self.covar_module(x)  # The covariance of the GP model is evaluated at the input data.

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)   # The GP model is evaluated at the input data and the mean and covariance are returned. 

def GPTrain(features, targets, meanPrior): # features: The input data for training the GP model. targets: The output data for training the GP model. meanPrior: The prior to use for the mean function of the GP.
    tensorSamplesXY = torch.from_numpy(features).double()  # Convert to DoubleTensor # The input data is converted to a PyTorch DoubleTensor.
    tensorSamplesZ = torch.from_numpy(targets).double().squeeze()  # Convert to DoubleTensor            # The output data is converted to a PyTorch DoubleTensor and the dimensions are squeezed.          

    likelihood = gpytorch.likelihoods.GaussianLikelihood().double() # The likelihood for the GP model is set to a Gaussian likelihood.
    model = ExactGPModel(tensorSamplesXY, tensorSamplesZ, likelihood, meanPrior).double()    # The GP model is initialized with the input data, output data, likelihood, and mean prior.       

    likelihood.noise = 1e-4     # The noise variance of the likelihood is set to 1e-4.
    likelihood.noise_covar.raw_noise.requires_grad_(False) # The noise variance of the likelihood is set to be not trainable.

    training_iter = 250 # The number of training iterations is set to 250.
    model.train()  # The GP model is set to training mode.
    likelihood.train() # The likelihood is set to training mode.

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05) # The Adam optimizer is used to optimize the parameters of the GP model.
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) # The marginal log likelihood is computed using the likelihood and the GP model.

    for i in range(training_iter): # The training loop is run for the specified number of iterations.
        optimizer.zero_grad()  # The gradients of the optimizer are set to zero.
        output = model(tensorSamplesXY) # The GP model is evaluated at the input data.
        loss = -mll(output, tensorSamplesZ) # The negative marginal log likelihood is computed.
        loss.backward() # The gradients are computed using backpropagation.
        optimizer.step() # The optimizer takes a step based on the gradients.

    return model # The trained GP model is returned.



def BOGPEval(model, newFeatures):   # model: The trained GP model. newFeatures: The input data for which the GP model is evaluated.
    model.eval()   # The GP model is set to evaluation mode.

    with torch.no_grad(), gpytorch.settings.fast_pred_var(): # The predictions are made without gradient tracking and with fast predictive variance computation.
        observed_pred = model(torch.from_numpy(newFeatures)) # The GP model is evaluated at the input data.

    mean_pred = observed_pred.mean.numpy() # The mean of the GP model predictions is extracted.
    stdDev = observed_pred.stddev.numpy() # The standard deviation of the GP model predictions is extracted.

    return mean_pred, stdDev # The mean and standard deviation of the GP model predictions are returned.


def expectedImprovement(feature, currentGP, bestY, epsilon): # feature: The input data for which the expected improvement is computed. currentGP: The trained GP model. bestY: The best observed value so far. epsilon: A small value to prevent division by zero.
    yPred, yStd = BOGPEval(currentGP, feature) # The GP model is evaluated at the input data and the mean and standard deviation of the predictions are extracted.


    #to maximise
    z = (yPred - bestY - epsilon) / yStd  # Adjusted for maximization
    ei = ((yPred - bestY - epsilon) * norm.cdf(z)) + yStd * norm.pdf(z)


    #to minimise
    #z = (bestY - yPred - epsilon) / yStd # The z-score is computed using the mean and standard deviation of the GP model predictions.
    #ei = ((bestY - yPred - epsilon) * norm.cdf(z)) + yStd * norm.pdf(z) # The expected improvement is computed using the z-score.
    return ei   # The expected improvement is returned.

def ei_wrapper(x, currentGP, bestY, epsilon): # x: The input data for which the expected improvement is computed. currentGP: The trained GP model. bestY: The best observed value so far. epsilon: A small value to prevent division by zero.
    # Reshape x if needed (differential_evolution passes a flat array)
    x = np.array(x).reshape(1, -1)   # The input data is reshaped to a 2D array.

    return -expectedImprovement(x, currentGP, bestY, epsilon)   # The negative expected improvement is computed and returned.

def optimize_ei(bounds, currentGP, bestY, epsilon): # bounds: The bounds for each dimension. currentGP: The trained GP model. bestY: The best observed value so far. epsilon: A small value to prevent division by zero.
    result = differential_evolution( # The differential evolution optimizer is used to optimize the expected improvement.
        ei_wrapper,                  # The wrapper function for the expected improvement
        bounds=bounds,                 # Bounds for each dimension
        args=(currentGP, bestY, epsilon),  # Additional arguments for the EI function
        strategy='best1bin',          # Default strategy
        maxiter=100,                  # Maximum number of iterations
        popsize=15,                   # Population size (of DE trial points)
        tol=1e-6,                     # Tolerance for convergence
        seed=None,                    # Random seed for reproducibility
    )
    return result


def blackbox(feature):
    camber, camber_loc, thickness = feature

    def create_airfoil(camber, camber_loc, thickness): #function to write Allrun File
                template_file = "generate_airfoil_template_2d.py" # Path to the template file
                salome_path = "/home/tomgorringe/Salome/SALOME-9.13.0-native-UB24.04-SRC/generate_airfoil.py" # Destination Path
                #Check Exists
                #camber = float(camber)

                try:
                    #camber = float(camber.item())

                    with open(template_file, 'r') as file:
                        gen_file_contents = file.read()
                    gen_file_contents = gen_file_contents.replace("$CAMBER", str(camber))
                    gen_file_contents = gen_file_contents.replace("$CLOC", str(camber_loc))
                    gen_file_contents = gen_file_contents.replace("$THICKNESS", str(thickness))
                    with open(salome_path, 'w') as file: # Write the updated contents to the new Allrun file in the destination directory
                        file.write(gen_file_contents)
                    os.chmod(salome_path, 0o755) # Set permissions to make it executable
                    print(f"Successfully created Generate Airfoil file at {salome_path}")
                except FileNotFoundError:
                        print(f"Error: Template file '{template_file}' not found.")
                        return
                except Exception as e:
                  print(f"Error writing to {salome_path}: {e}")

    
    
    cam = float(camber)
    loc = float(camber_loc)
    thick = float(thickness)
    print(f"Camber = {cam} Location = {loc} Thickness = {thick}")
    create_airfoil(cam,loc,thick)
    salome_shell = "/home/tomgorringe/Salome/SALOME-9.13.0-native-UB24.04-SRC/salome"
    script_path = "/home/tomgorringe/Salome/SALOME-9.13.0-native-UB24.04-SRC/generate_airfoil.py"

    subprocess.run(["bash", "-c", f"{salome_shell} -t {script_path}"], check=True)

    original_case = "/home/tomgorringe/OpenFOAM/tomgorringe-v2312/run/optimisation_airfoil/blank_case"
    src = original_case
    dst = "/home/tomgorringe/OpenFOAM/tomgorringe-v2312/run/optimisation_airfoil/new_case"

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

    time.sleep(10)

    stl_generated_path = "/home/tomgorringe/airfoil.stl"
    stl_output_path = "/home/tomgorringe/OpenFOAM/tomgorringe-v2312/run/optimisation_airfoil/new_case/constant/triSurface"

    if os.path.exists(stl_generated_path):
        shutil.copy(stl_generated_path, stl_output_path)
        print(f"STL file copied to: {stl_output_path}")
        os.remove(stl_generated_path)
    else:
        print(f"STL file not found at {stl_generated_path}")
    print("Starting Allrun")


    try:
        subprocess.run(["bash", "run_allrun.bashrc"], check=True)
        print(f"Ran Allrun. Check log for more details.")
    except Exception as e:
        print(f"Error executing the Allrun script: {e}")    


    def read_cl_cd(file_name):
        pth = "/home/tomgorringe/OpenFOAM/tomgorringe-v2312/run/optimisation_airfoil/"
        case = file_name
        file_path = os.path.join(pth,case, 'postProcessing', 'forceCoeffs1', '0', 'coefficient.dat')

         #Regex to match the force coefficient data
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
    print(f"Camber: {cam:.3f}")   
    print(f"Location: {loc:.3f}")      
    print(f"Thickness: {thick:.3f}")      
    print(f"Lift-to-Drag Ratio (L/D): {L_D_ratio:.3f}")

    return L_D_ratio

def whitebox (feature):
    a,b = feature
    a = float(a)
    b = float(b)
    LD = 20*a**2+64*b**0.2
    return LD
    

lowerBound = [0, 0.3, 0.3] # The lower bound of the input data is set to -10.
upperBound = [0.2, 0.8, 0.8] # The upper bound of the input data is set to 10.

initialSampleSize = 3 # The initial sample size is set.
print("Setting Bounds")

sampler = qmc.LatinHypercube(d=3) # LHS initialised and number of dimensions is set.
 
sample = sampler.random(n=initialSampleSize) # The Latin Hypercube sampler is used to generate a random sample of the specified size.
features = np.array(qmc.scale(sample, lowerBound, upperBound)) # The random sample is scaled to the specified bounds and converted to a NumPy array.
print("LHS Completed, Starting Target Computation")
print(features)

targets = np.zeros((len(features), 1))

for idx,i in enumerate(features):
    ix = idx+1
    print(f"Sample Simulation {ix}/{len(features)}. Max Camber = {i[0]} Location = {i[1]} Thickness = {i[2]}")
    targets[idx] = blackbox(i)
    print("Current Features:")
    print(features)
    print("Current Targets:")
    print(targets)

x_range = np.linspace(lowerBound, upperBound, 100) # The range of the input data is set.


destpng = '/home/tomgorringe/pngfolder'

# print("Printing LHS Results")
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(features[:, 0], features[:, 1], targets, color='red', label='Samples')
# ax.set_xlabel('Camber')
# ax.set_ylabel('Location')
# ax.set_zlabel('L/D Ratio')
# ax.set_title('Sampled Points in 3D')
# ax.legend()
# destlhs = os.path.join(destpng, "1lhsresults_3d.png")
# plt.savefig(destlhs, dpi=300)

globalGP = GPTrain(features, targets, meanPrior="max") 

y_pred, y_std = BOGPEval(globalGP, x_range)

best_idx = np.argmax(targets) # The index of the best observed value is computed.
best_x = features[best_idx] # The input data corresponding to the best observed value is extracted.
best_y = targets[best_idx] # The best observed value is extracted.

lowerBound1, upperBound1 = 0.0, 0.3  # First dimension
lowerBound2, upperBound2 = 0.2, 0.8  # Second dimension
lowerBound3, upperBound3 = 0.2, 0.8  # Third dimension

# Set up the nested bounds as tuples
bounds = [(lowerBound1, upperBound1), (lowerBound2, upperBound2), (lowerBound3, upperBound3)]

print("bounds=",bounds)
result = optimize_ei(bounds, globalGP, best_y, 0.1) # The expected improvement is optimized using differential evolution.


optimal_feature = result.x # The optimal feature that maximizes the expected improvement is extracted.
max_ei = -ei_wrapper(optimal_feature, globalGP, best_y, 0.1) # The maximum expected improvement is computed at the optimal feature.

ei = expectedImprovement(x_range, globalGP, best_y, 0.1)

iteration = 0


while np.max(ei) > 1e-5:
    iteration += 1
    print(f"Iteration {iteration}")
    x_range = np.linspace(lowerBound, upperBound, 100)
    globalGP = GPTrain(features, targets, meanPrior="max")
    
    y_pred, y_std = BOGPEval(globalGP, x_range) 
    best_idx = np.argmax(targets)
    best_x = features[best_idx]
    best_y = targets[best_idx]
    print("Current Best It:",best_idx,"Best Features",best_x,"Best L/D",best_y)
    
    ei = expectedImprovement(x_range, globalGP, best_y, 0.1)  
    
    result = optimize_ei(bounds, globalGP, best_y, 0.1)  
    
    new_x = result.x  # Reshape new_x to match features
    print("new x =",new_x)
    
    max_ei = -ei_wrapper(new_x, globalGP, best_y, 0.1)
    
    new_y = blackbox(new_x)
    
    features = np.vstack([features, new_x])  
    targets = np.append(targets, new_y)

    print("Features")
    print(features)
    print("Targets")
    print(targets.reshape(-1, 1))
    
    # ei = expectedImprovement(x_range, globalGP, best_y, 0.1)

    # x_range = np.linspace(0,0.3,100)
    
    # y_range = np.linspace(0.2,0.8,100)

    # fullRange = list(product(x_range, y_range))
    # fullRangeArray = np.array(fullRange)
    # rangePredictions, rangeStdev = BOGPEval(globalGP, fullRangeArray)

    # fig = plt.figure(figsize=(12, 10))

    # plt.scatter(fullRangeArray[:,0], fullRangeArray[:,1], c = rangePredictions)
    # sc = plt.scatter(fullRangeArray[:,0], fullRangeArray[:,1], c = rangePredictions)
    # plt.scatter(features[:,0], features[:,1], c='blue', marker='X', label='Dataset' )
    # plt.colorbar(sc, label="L/D Values")
    # plt.legend()
    # #plt.show()

    # destit = os.path.join(destpng,f"iteration{iteration}.png")
    # plt.savefig(destit, dpi=300)
    # print("Plots Printed")



print("Converged!")
print()

x_range = np.linspace(0,0.3,0.3,100)
    
y_range = np.linspace(0.2,0.8,0.8,100)

fullRange = list(product(x_range, y_range))
fullRangeArray = np.array(fullRange)
rangePredictions, rangeStdev = BOGPEval(globalGP, fullRangeArray)

fig = plt.figure(figsize=(12, 10))

plt.scatter(fullRangeArray[:,0], fullRangeArray[:,1], c = rangePredictions)
plt.scatter(features[:,0], features[:,1], c='blue', marker='X', label='Dataset' )
plt.colorbar()
plt.legend()
plt.show()

plt.show()
destconv = os.path.join(destpng, "zConverged.png")
plt.savefig(destconv, dpi=300)
plt.show()
print("Converged Plot Printed")


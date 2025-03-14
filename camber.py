import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from scipy.stats import norm
import subprocess
import shutil
import os
import sys
import time
import re


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


    z = (bestY - yPred - epsilon) / yStd # The z-score is computed using the mean and standard deviation of the GP model predictions.
    ei = ((bestY - yPred - epsilon) * norm.cdf(z)) + yStd * norm.pdf(z) # The expected improvement is computed using the z-score.
    return ei   # The expected improvement is returned.


def blackbox(feature):

    def create_airfoil(camber): #function to write Allrun File
                template_file = "generate_airfoil_template.py" # Path to the template file
                salome_path = "/home/tomgorringe/Salome/SALOME-9.13.0-native-UB24.04-SRC/generate_airfoil.py" # Destination Path
                #Check Exists
                #camber = float(camber)
                try:
                    #camber = float(camber.item())
                    with open(template_file, 'r') as file:
                        gen_file_contents = file.read()
                    gen_file_contents = gen_file_contents.replace("$CAMBER", str(camber))# Replace the placeholder with the actual value
                    with open(salome_path, 'w') as file: # Write the updated contents to the new Allrun file in the destination directory
                        file.write(gen_file_contents)
                    os.chmod(salome_path, 0o755) # Set permissions to make it executable
                    print(f"Successfully created Generate Airfoil file at {salome_path}")
                except FileNotFoundError:
                        print(f"Error: Template file '{template_file}' not found.")
                        return
                except Exception as e:
                  print(f"Error writing to {salome_path}: {e}")

    
    
    cam = float(feature)
    print(f"Camber = {cam}")
    create_airfoil(cam)
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

        # Regex to match the force coefficient data
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
    print(f"Lift Coefficient (Cl): {cl}")
    print(f"Drag Coefficient (Cd): {cd}")

    
    L_D_ratio = cl / cd if cd != 0 else float("inf")
    print(f"Camber: {cam:.3f}")        
    print(f"Lift-to-Drag Ratio (L/D): {L_D_ratio:.3f}")

    return L_D_ratio


lowerBound = 0 # The lower bound of the input data
upperBound = 0.2 # The upper bound of the input data

initialSampleSize = 10
print("Setting Bounds")

sampler = qmc.LatinHypercube(
    d=1 # The number of dimensions
)  
sample = sampler.random(n=initialSampleSize) # The Latin Hypercube sampler is used to generate a random sample of the specified size.
features = np.array(qmc.scale(sample, lowerBound, upperBound)) # The random sample is scaled to the specified bounds and converted to a NumPy array.
print("LHS Completed, Starting Target Computation")
print(features)

targets = np.zeros((len(features), 1))

for idx,i in enumerate(features):
    ix = idx+1
    print(f"Sample Simulation {ix}/{len(features)}. Max Camber = {i}")
    targets[idx] = blackbox(i)
    print("Current Cambers:")
    print(features)
    print("Current L/Ds:")
    print(targets)

x_range = np.linspace(lowerBound, upperBound, 100) # The range of the input data is set

destpng = '/home/tomgorringe/pngfolder'

print("Printing LHS Results")
plt.scatter(features, targets, color='red', label='Samples')
plt.xlabel('x')
plt.ylabel('Output')
plt.title('Sampled Points')
plt.legend()
destlhs = os.path.join(destpng,"1lhsresults.png")
plt.savefig(destlhs, dpi=300)
#plt.show()


globalGP = GPTrain( # The GP model is trained with the input data, output data, and the mean prior.
    features, targets, meanPrior="max" # The mean prior is set
) 

y_pred, y_std = BOGPEval(globalGP, x_range) # The GP model is evaluated at the range of input data and the mean and standard deviation of the predictions are extracted.

print("Printing GP Model")
plt.figure(figsize=(10, 6))
plt.scatter(features, targets, color='red', label='Samples')
plt.plot(x_range, y_pred, color='blue', label='Gaussian Process')
plt.fill_between(x_range, y_pred - 2*y_std, y_pred + 2*y_std, color='blue', alpha=0.2)
plt.xlabel('x')
plt.ylabel('Black Box Output')
plt.title('Black Box Function with Gaussian Process Surrogate Model')
plt.legend()
destgp = os.path.join(destpng,"2gpmodel.png")
plt.savefig(destgp, dpi=300)
#plt.show()

best_idx = np.argmax(targets) # The index of the best observed value is computed.
best_x = features[best_idx] # The input data corresponding to the best observed value is extracted.
best_y = targets[best_idx] # The best observed value is extracted.

print("Printing EI")

ei = expectedImprovement(x_range, globalGP, best_y, 0.1) # The expected improvement is computed at the range of input data.
plt.figure(figsize=(10, 6)) 
plt.plot(x_range, ei, color='green', label='Expected Improvement')
plt.xlabel('x')
plt.ylabel('Expected Improvement')
plt.title('Expected Improvement')
plt.legend()
destei1 = os.path.join(destpng,"3expectedimprovement.png")
plt.savefig(destei1, dpi=300)
#plt.show()


# differential evolution optimisation of expected improvement
# for scipy to handle this we need a wrapper function:
# Wrapper function for expected improvement
def ei_wrapper(x, currentGP, bestY, epsilon): # x: The input data for which the expected improvement is computed. currentGP: The trained GP model. bestY: The best observed value so far. epsilon: A small value to prevent division by zero.
    # Reshape x if needed (differential_evolution passes a flat array)
    x = np.array(x).reshape(1, -1)   # The input data is reshaped to a 2D array.
    return -expectedImprovement(x, currentGP, bestY, epsilon)   # The negative expected improvement is computed and returned.


# Function to optimize EI using differential evolution
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


bounds = [(lowerBound, upperBound)] # The bounds for each dimension are set to the lower and upper bounds.
result = optimize_ei(bounds, globalGP, best_y, 0.1) # The expected improvement is optimized using differential evolution.

# Optimal feature that maximizes EI
optimal_feature = result.x # The optimal feature that maximizes the expected improvement is extracted.
max_ei = -ei_wrapper(optimal_feature, globalGP, best_y, 0.1) # The maximum expected improvement is computed at the optimal feature.

print("Printing EI")
plt.figure(figsize=(10, 6))
plt.plot(x_range, ei, color='green', label='Expected Improvement')
plt.scatter(optimal_feature, max_ei)
plt.xlabel('x')
plt.ylabel('Expected Improvement')
plt.title('Expected Improvement')
plt.legend()
destei2 = os.path.join(destpng,"4expectedimprovement2.png")
plt.savefig(destei2, dpi=300)
#plt.show()

iteration = 0  # The iteration counter is initialized to 0.

while np.max(ei) > 1e-10: # The optimization loop continues until the maximum expected improvement is less than 1e-5.
        iteration += 1 # The iteration counter is incremented.
        print(f"Iteration {iteration}")
        # Fit the Gaussian process model to the sampled points
        globalGP = GPTrain( # The GP model is trained with the input data, output data, and the mean prior.
            features, targets, meanPrior="max" # The mean prior is set to "zero".
        )
        y_pred, y_std = BOGPEval(globalGP, x_range) # The GP model is evaluated at the range of input data and the mean and standard deviation of the predictions are extracted.
        # Determine the point with the lowest observed function value
        best_idx = np.argmin(targets) # The index of the best observed value is computed.
        best_x = features[best_idx] # The input data corresponding to the best observed value is extracted.
        best_y = targets[best_idx] # The best observed value is extracted.

        ei = expectedImprovement(x_range, globalGP, best_y, 0.1)    # The expected improvement is computed at the range of input data.


        # Plot the black box function, surrogate function, previous points, and new points
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize = (10,10)) 
        plt.subplot(2,1,1)
    
        plt.plot(x_range, y_pred, color='red', linestyle='dashed', label='Surrogate Function')
        plt.fill_between(x_range, y_pred - 2*y_std, y_pred + 2*y_std, color='blue', alpha=0.2)
        plt.scatter(features, targets, color='blue', label='Previous Points')
        plt.legend()
        plt.title(f"Iteration #{iteration}") 
        
        result = optimize_ei(bounds, globalGP, best_y, 0.1) #

        # Optimal feature that maximizes EI 
        new_x = result.x 
        max_ei = -ei_wrapper(new_x, globalGP, best_y, 0.1)     

        new_y = blackbox(new_x)
        features = np.append(features, new_x)
        targets = np.append(targets, new_y)
        plt.scatter(new_x, new_y, color='green', label='New Points')

        plt.subplot(2,1,2)
        plt.plot(x_range, ei, color='green', label='Expected Improvement')
        plt.scatter(new_x, max_ei)

        plt.xlabel('x')
        plt.ylabel('Expected Improvement')
        plt.title('Expected Improvement')
        plt.ylabel('y')
        plt.legend()
        
        destit = os.path.join(destpng,f"iteration{iteration}.png")
        plt.savefig(destit, dpi=300)
        print("Plots Printed")
        #plt.show()

plt.figure(figsize=(10, 5))  # Set figure size

plt.plot(x_range, y_pred, color='red', linestyle='dashed', label='Surrogate Function')
plt.fill_between(x_range, y_pred - 2*y_std, y_pred + 2*y_std, color='blue', alpha=0.2)
plt.scatter(features, targets, color='blue', label='Previous Points')

plt.legend()
plt.title("Converged")
plt.xlabel("X-axis Label")  # Optional
plt.ylabel("Y-axis Label")  # Optional
destconv = os.path.join(destpng,"zConvereged.png")
plt.savefig(destconv, dpi=300)

print('Converged!')
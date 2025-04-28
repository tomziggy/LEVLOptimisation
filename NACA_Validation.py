import numpy as np
import torch
from scipy.stats import qmc
import os
import csv
import time
import re
import subprocess
import shutil

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

def MTCHScalarisation(targets, weights=None, a=0.1):
    cl = targets[:, 0]
    cdinv = targets[:, 1]
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

# ---------------------------------------------------------
# Main Monte Carlo Simulation
# ---------------------------------------------------------

start = time.time()

airfoil_designs = np.array([

    [0.05618102, 0.77042858, 0.18567855]


])

results = []
# Set bounds
lowerBound = [0, 0.2, 0.01]
upperBound = [0.15, 0.8, 0.25]
clowerBound = [0, 5]  # AOA and Velocity lower bounds
cupperBound = [20, 25]  # AOA and Velocity upper bounds

initialSampleSize = 10
cinitialSampleSize = 5
alpha = 0.1
runningdirectory = '/home/tomgorringe/'

# Sample conditions
# print("Sampling Conditions...")
# csampler = qmc.LatinHypercube(d=len(clowerBound))
# csample = csampler.random(n=cinitialSampleSize)
# conditions = np.array(qmc.scale(csample, clowerBound, cupperBound))
conditions = np.array([
    [12.47851984, 8.19605446]
])
print("Conditions:")
print(conditions)

# Full Monte Carlo over features
for i, design in enumerate(airfoil_designs):
    for j, cond in enumerate(conditions):
        # Build input [camber, camber_location, thickness, aoa, velocity]
        input_row = np.concatenate([design, cond])
        print(f"\n🔧 Simulating Design {i+1}, Condition {j+1}: {input_row}")

        # Call blackbox
        cl, cd_inv = blackbox(input_row)
        cd = 1 / cd_inv if cd_inv != 0 else float("inf")
        cl_cd = cl / cd if cd != 0 else float("inf")

        # Store
        results.append([
            i+1, j+1,
            design[0], design[1], design[2],
            cond[0], cond[1],
            cl, cd, cl_cd, cd_inv
        ])

# === Save results to CSV ===
results_dir = os.path.join(runningdirectory, "results")
os.makedirs(results_dir, exist_ok=True)

csv_path = os.path.join(results_dir, "specific_airfoil_simulations.csv")

headers = [
    "Design#", "Condition#",
    "Camber", "Camber Location", "Thickness",
    "AOA", "Velocity",
    "Cl", "Cd", "Cl/Cd", "1/Cd"
]

with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(results)

print(f"\n✅ All simulations complete! Results saved to {csv_path}")
print(f"🕒 Total time: {(time.time() - start):.2f} seconds")

# === Optional: Pretty print results to console ===
print("\n📄 Summary:")
for row in results:
    print(f"Design {row[0]}, Condition {row[1]} | Camber={row[2]:.4f} | Loc={row[3]:.4f} | Thick={row[4]:.4f} | AOA={row[5]} | U={row[6]} || Cl={row[7]:.4f}, Cd={row[8]:.4f}, Cl/Cd={row[9]:.4f}, 1/Cd={row[10]:.4f}")
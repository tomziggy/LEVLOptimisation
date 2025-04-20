# LEVLOptimisation
Repo for optimisation process for LEVL Probe

# Step 1 - Install WSL
https://learn.microsoft.com/en-us/windows/wsl/install

# Step 2 - Install OpenFOAM v2312
Make sure its v2312 not another version or OpenFOAM 10 for KSEEGER model
https://www.youtube.com/watch?v=CeEJS1eT9NE

# Step 3 - Install Salome
Make sure correct Linux Distro and Version
https://www.salome-platform.org/

# Step 4 - Setup Directories
Replace user
```
/home/user
│── OpenFOAM/                
│   ├── user-v2312/              
│       ├── run/
│           ├── optimisation_airfoil/
│               └── blank_case/
│
│── SALOME/       
│   ├── SALOME-9.13.0-native-UB24.04-SRC/
│
│── RMO_UNIVERSAL.py
│
│── forceCoeffs_template
│
│── U_template
│
│── generate_airfoil_template_3d.py
│
│── run_allrun.bashrc
│
│── results/
```

# Step 5 - Edit RMO Scripts
To run through your directories

# Step 6 - Check Salome and OpenFOAM
In their directories type 'salome' or 'openfoam232'

# Step 7 - Install Req Python Packages
numpy
matplotlib.pyplot
torch
gpytorch
scipy
mpl_toolkits.mplot3d
subprocess
shutil
os
sys
time
re
itertools
csv

# Step 8 - Run
python3 RMO_Universal_2D

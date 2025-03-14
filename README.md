# LEVLOptimisation
Repo for optimisation process for LEVL Probe

# Step 1 - Install WSL
https://learn.microsoft.com/en-us/windows/wsl/install

# Step 2 - Install OpenFOAM v2312
Make sure its v2312 not another version or OPenFOAM 10 for KSEEGER model
https://www.youtube.com/watch?v=CeEJS1eT9NE

# Step 3 - Install Salome
Make sure correct Linux Distro and Version
https://www.salome-platform.org/

# Step 4 - Setup Directories
Replace tomgorringe with user
'''
/home/tomgorringe
│── OpenFOAM/                
│   ├── tomgorringe-v2312/              
│       ├── run/
│           ├── optimisation_airfoil/
│               └── blank_case/
│
│── SALOME/       
│   ├── SALOME-9.13.0-native-UB24.04-SRC/
│
│── camber.py
│
│── twodim.py
│
│── generate_airfoil_template.py
│
│── run_allrun.bashrc
│
│── pngfolder/
'''

# Step 5 - Edit Scripts
To run through your directories rather than

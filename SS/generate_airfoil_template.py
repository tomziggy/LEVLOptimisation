import salome
from salome.geom import geomBuilder
import numpy as np

salome.salome_init()
geompy = geomBuilder.New()

def generate_naca_airfoil(m: float, p: float, t: float, num_points: int = 100):
    """
    Generate coordinates for a NACA 4-digit airfoil.

    Parameters:
    m (float): Maximum camber (as a fraction of chord length)
    p (float): Location of maximum camber (as a fraction of chord length)
    t (float): Maximum thickness (as a fraction of chord length)
    num_points (int): Number of points to generate along the chord

    Returns:
    list: List of (x, y) coordinates defining the airfoil
    """
    x_coords = np.linspace(0, 1, num_points)
    yt = 5 * t * (0.2969 * np.sqrt(x_coords) - 0.1260 * x_coords -
                  0.3516 * x_coords ** 2 + 0.2843 * x_coords ** 3 - 
                  0.1015 * x_coords ** 4)

    yc = np.where(x_coords <= p,
                   m / p ** 2 * (2 * p * x_coords - x_coords ** 2),
                   m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x_coords - x_coords ** 2))

    dyc_dx = np.where(x_coords <= p,
                      2 * m / p ** 2 * (p - x_coords),
                      2 * m / (1 - p) ** 2 * (p - x_coords))

    theta = np.arctan(dyc_dx)
    xu = x_coords - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x_coords + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    upper_surface = list(zip(xu, yu))
    lower_surface = list(zip(xl[::-1], yl[::-1]))

    return upper_surface + lower_surface


def create_fluid_domain(airfoil_coords, domain_size=(10, 10, 0.01)):
    """
    Create the fluid domain (bounding box) around the airfoil.

    Parameters:
    airfoil_coords (list): List of (x, y) points defining the airfoil.
    domain_size (tuple): Size of the fluid domain (width, length, height).
    
    Returns:
    object: Salome geometry object for the fluid domain (box).
    """
    
    x_min = min([x for x, _ in airfoil_coords])
    x_max = max([x for x, _ in airfoil_coords])
    y_min = min([y for _, y in airfoil_coords])
    y_max = max([y for _, y in airfoil_coords])

    
    padding = 1
    x_min -= padding
    x_max += 2*padding
    y_min -= padding
    y_max += padding
    
    
    fluid_box = geompy.MakeBoxDXDYDZ(x_max - x_min, y_max - y_min, domain_size[2])
    translated_fluid_box = geompy.MakeTranslation(fluid_box, x_min, y_min, 0)

    return translated_fluid_box

def create_airfoil_and_fluid_domain(airfoil_coords):
    """
    Create the airfoil geometry and the fluid domain, then subtract the airfoil
    from the domain to create a cavity for the fluid simulation.

    Parameters:
    airfoil_coords (list): List of (x, y) points defining the airfoil.
    """
    
    points = [geompy.MakeVertex(x, y, 0) for x, y in airfoil_coords]
    spline = geompy.MakeInterpol(points)
    airfoil_surface = geompy.MakeFaceWires([spline], True)
    
   
    extrusion_height = 0.1 
    airfoil_extrusion = geompy.MakePrismVecH(airfoil_surface, geompy.MakeVectorDXDYDZ(0, 0, 1), extrusion_height)

    fluid_domain = create_fluid_domain(airfoil_coords)

    fluid_with_airfoil_removed = geompy.MakeBoolean(fluid_domain, airfoil_extrusion,2)
    
    geompy.addToStudy(fluid_with_airfoil_removed, f"Fluid Domain with Airfoil Cavity")
    geompy.addToStudy(airfoil_extrusion, f"Airfoil Geometry")
    geompy.addToStudy(fluid_domain, f"Fluid Domain")
    geompy.addToStudy(airfoil_surface, f"Airfoil Surface")

    geompy.ExportSTL(airfoil_extrusion, "airfoil.stl")

   
    #salome.sg.updateObjBrowser()

f = float("$CAMBER")
print("Camber Inputted")
airfoil_coords = generate_naca_airfoil(m=f, p=0.4, t=0.12)
print("Coords Generated")
create_airfoil_and_fluid_domain(airfoil_coords)
print("Geometry Generated")

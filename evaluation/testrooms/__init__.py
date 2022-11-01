import pywavefront
import os

script_path = os.path.dirname(__file__)

CUBE = pywavefront.Wavefront(os.path.join(script_path, "models/cube.obj"))
BIG_CUBE = pywavefront.Wavefront(os.path.join(script_path, "models/big_cube.obj"))
COMPLEX_HOUSE = pywavefront.Wavefront(os.path.join(script_path, "models/complex_house.obj"))
CORRIDOR = pywavefront.Wavefront(os.path.join(script_path, "models/corridor.obj"))
HAUS_DES_NIKOLAUS= pywavefront.Wavefront(os.path.join(script_path, "models/haus_des_nikolaus.obj"))
SLOPE = pywavefront.Wavefront(os.path.join(script_path, "models/slope.obj"))
TELEVISION = pywavefront.Wavefront(os.path.join(script_path, "models/television.obj"))
WINDOW = pywavefront.Wavefront(os.path.join(script_path, "models/window.obj"))
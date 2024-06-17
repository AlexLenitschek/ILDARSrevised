import pywavefront
import os

script_path = os.path.dirname(__file__)

CUBE = pywavefront.Wavefront(os.path.join(script_path, "models/cube.obj"), collect_faces=True)
BIG_CUBE = pywavefront.Wavefront(os.path.join(script_path, "models/big_cube.obj"), collect_faces=True)
COMPLEX_HOUSE = pywavefront.Wavefront(os.path.join(script_path, "models/complex_house.obj"), collect_faces=True)
CORRIDOR = pywavefront.Wavefront(os.path.join(script_path, "models/corridor.obj"), collect_faces=True)
HAUS_DES_NIKOLAUS= pywavefront.Wavefront(os.path.join(script_path, "models/haus_des_nikolaus.obj"), collect_faces=True)
SLOPE = pywavefront.Wavefront(os.path.join(script_path, "models/slope.obj"), collect_faces=True)
TELEVISION = pywavefront.Wavefront(os.path.join(script_path, "models/television.obj"), collect_faces=True)
WINDOW = pywavefront.Wavefront(os.path.join(script_path, "models/window.obj"), collect_faces=True)
TEST1ROOM = pywavefront.Wavefront(os.path.join(script_path, "models/test1room.obj"), collect_faces=True)
PYRAMIDROOM = pywavefront.Wavefront(os.path.join(script_path, "models/pyramidroom.obj"), collect_faces=True)
CONCERTHALL = pywavefront.Wavefront(os.path.join(script_path, "models/concerthall.obj"), collect_faces=True)
TEST1ROOMNEW = pywavefront.Wavefront(os.path.join(script_path, "models/concerthall.obj"), collect_faces=True)
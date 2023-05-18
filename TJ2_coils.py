
from simsopt.geo import CurveXYZFourier, curves_to_vtk
from simsopt.geo import SurfaceRZFourier
from simsopt.field import coils_via_file
from simsopt.field.biotsavart import BiotSavart
import numpy as np

curves = CurveXYZFourier.load_curves_from_file("coils.tj2.in.amperes.1004464",100)

coils = coils_via_file("coils.tj2.in.amperes.1004464",100)


curves_to_vtk(curves, "TJ2_coils", close=True)

surface_filename = "input.100_44_64_0.0"
s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=64, ntheta=64)
s.to_vtk("TJ2_equilibrium")

bs = BiotSavart(coils)
bs.save("TJ-II_coils.json")
# Set up correct Bnormal from TF coils 
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((64, 64, 3)) * s.unitnormal(), axis=2)

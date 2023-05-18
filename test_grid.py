import os
from pathlib import Path
import numpy as np
from simsopt.geo import  SurfaceRZFourier, curves_to_vtk, Surface, CurveRZFourier, PermanentMagnetGrid
import simsoptpp as sopp
from simsopt.mhd.vmec import Vmec
from simsopt import load
from simsopt.util.permanent_magnet_helper_functions import *

# Set some parameters
comm = None
nphi = 64 # need to set this to 64 for a real run
ntheta = 64 # same as above
#dr = 0.02  #dr is used when using cylindrical coordinates
Nx = 10     #Nx is used when using cartesian coordinates
surface_flag = 'vmec'
#input_name = 'input.100_44_64_0.0_000_000000'
input_name = 'wout_100_44_64_0.0_000_000000.nc'
preciseQH_wout_name = 'wout_preciseQH_rescaled_TJ-II_PHIEDGE=0.096.nc'
coordinate_flag = 'cartesian'  #I am using cartesian as the orientation for the magnets is only implemented in the cartesian case.

# Make the output directory
OUT_DIR = './test_grid/'
os.makedirs(OUT_DIR, exist_ok=True)

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent).resolve()
surface_filename = str(TEST_DIR / input_name)
s_eq_current = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta) #current TJ-II eq
s_eq_current.to_vtk(OUT_DIR + "eq_surface_TJ-II")

s_eq_preciseQH = SurfaceRZFourier.from_wout(str(TEST_DIR/preciseQH_wout_name), range="half period", nphi=nphi, ntheta=ntheta)
s_eq_preciseQH.to_vtk(OUT_DIR + "eq_preciseQH_TJ-II")

#setting radius for the circular coils
vmec = Vmec(surface_filename)
#vmec.run()
# Number of Fourier modes describing each Cartesian component of each coil:
order = 5

coilfile = "TJ-II_coils.json" #I converted the TJ-II input into a json through some functions I have made that allow uploading makegrid files to simsopt (pull request open atm)
bs = load(coilfile)
coils = bs.coils
ncoils = len(coils)
base_curves = [coils[i].curve for i in range(ncoils)]
base_currents = [coils[i].current for i in range(ncoils)]

# Set up correct Bnormal from TF coils 
bs.set_points(s_eq_preciseQH.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s_eq_preciseQH.unitnormal(), axis=2)

#find magnetic axis
ma = CurveRZFourier(np.linspace(0,1./(2*s_eq_current.nfp),25*10, endpoint=False), 24, nfp=s_eq_current.nfp, stellsym=True)

ma.rc[:] = [1.50617197940474, 0.229677578953876, 0.00244375813601314, 
    0.000902887527223983, -4.77998400571652e-05, 0.000235470539888026, 
    6.24755661224816e-05, -9.68521401464357e-05, 2.13758998983309e-05, 
    0.000104806094196373, 1.83783585816582e-05, 6.55537462967682e-07, 
    -1.11886035336001e-06, -2.5756136476815e-06, -1.7945202587065e-06, 
    -1.50278334318679e-06, -5.90525837920686e-08, 2.3800982274042e-07, 
    1.16947427672752e-08, -6.89744909143449e-08, -4.59322093316232e-08, 
    -2.09356659169718e-08, -8.32165425379089e-09, -1.12711282920907e-08, 
    7.83086269139466e-09]

ma.zs[:] = -np.array([ 0.23298323722167, 0.000259029950179007, 0.00025586143207216, 
    0.000108219073179918, -2.24859242478245e-05, 0.000175919044617549, 
    0.000118902651514014, -2.79391067517109e-05, 8.51655167855847e-05, 
    1.02685108635267e-05, -1.16972230376642e-06, -2.66202301183455e-06, 
    2.00771346066938e-06, -2.47203625254048e-06, -1.33121443305225e-06, 
    -6.70353078490061e-09, 2.16121417985215e-07, 4.37659412448007e-08, 
    -9.71242921266914e-08, -6.42276860626212e-08, -1.60818360982255e-08, 
    1.1447855734691e-09, -1.23750576608888e-08, -1.98598041013974e-08])


curves_to_vtk([ma],OUT_DIR + "magnetic_axis")

#create inside surface
s_in = SurfaceRZFourier(ntor=nphi, mpol=int(ntheta/2), nfp = s_eq_current.nfp) #current TJ-II eq
s_in.fit_to_curve(ma, 0.6, flip_theta=True)
s_in.to_vtk(OUT_DIR + "surface_in")

#create outside surface
s_out = SurfaceRZFourier(ntor=nphi, mpol=ntheta, nfp = s_eq_current.nfp) #current TJ-II eq
s_out.fit_to_curve(ma, 0.8, flip_theta=True)
s_out.to_vtk(OUT_DIR + "surface_out")


##Initialize the permanent magnet class
pm_opt = PermanentMagnetGrid(
    s_eq_preciseQH, rz_inner_surface=s_in, rz_outer_surface=s_out,
    Bn=Bnormal,
    Nx=Nx,
    coordinate_flag=coordinate_flag,
)

pm_opt.geo_setup()

print('Number of available dipoles = ', pm_opt.ndipoles)

write_pm_optimizer_to_famus(OUT_DIR, pm_opt)

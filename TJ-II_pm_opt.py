import os
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from simsopt.geo import SurfaceRZFourier, curves_to_vtk, CurveRZFourier, PermanentMagnetGrid
from simsopt.field import Current, Coil, BiotSavart, DipoleField
import simsoptpp as sopp
from simsopt.mhd.vmec import Vmec
from simsopt import load
from simsopt.util.permanent_magnet_helper_functions import *
from simsopt.solve import GPMO
from simsopt.objectives import SquaredFlux

# Set some parameters
nphi = 64 # need to set this to 64 for a real run
ntheta = 64 # same as above
dr = 0.03  #dr is used when using cylindrical coordinates
#Nx = 10     #Nx is used when using cartesian coordinates
surface_flag = 'vmec'
TJ_II_wout = 'wout_100_44_64_0.0_000_000000.nc'
preciseQH_wout_name = 'wout_preciseQH_rescaled_TJ-II_PHIEDGE=0.096.nc'
coordinate_flag = 'cylindrical'  

# Make the output directory
OUT_DIR = './TJ-II_PM_opt/'
os.makedirs(OUT_DIR, exist_ok=True)

# Read in the plasma equilibrium file
TEST_DIR = (Path(__file__).parent).resolve()
surface_filename = str(TEST_DIR/preciseQH_wout_name)

s_eq_preciseQH = SurfaceRZFourier.from_wout(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
s_eq_preciseQH.to_vtk(OUT_DIR + "eq_preciseQH_TJ-II")

# Read in the TJ-II equilibrium file
s_eq_current = SurfaceRZFourier.from_wout(str(TEST_DIR / TJ_II_wout), range="half period", nphi=nphi, ntheta=ntheta) #current TJ-II eq
s_eq_current.to_vtk(OUT_DIR + "eq_surface_TJ-II")

# Number of Fourier modes describing each Cartesian component of each coil:
order = 5

coilfile = "TJ-II_coils.json" #I converted the TJ-II input into a json through some functions I have made that allow uploading makegrid files to simsopt (pull request open atm)
bs = load(coilfile)
coils = bs.coils
ncoils = len(coils)
base_curves = [coils[i].curve for i in range(ncoils)]
#base_currents = [coils[i].current for i in range(ncoils)] #in case you wish to use the original currents.
base_currents = [Current(1.0) * 1e5 for i in range(ncoils)]

coils_to_opt = []
for i in range(ncoils):
    coils_to_opt.append(Coil(base_curves[i], base_currents[i]))
    
base_currents[0].fix_all() #fix one of the currents to they dont go to 0

# fix all the coil shapes so only the currents are optimized
for i in range(ncoils):
    base_curves[i].fix_all()
    
curves = [c.curve for c in coils_to_opt]

# Set up BiotSavart fields
bs = BiotSavart(coils_to_opt)

# Make higher resolution surface for plotting Bnormal
qphi = 2 * nphi
quadpoints_phi = np.linspace(0, 1, qphi, endpoint=True)
quadpoints_theta = np.linspace(0, 1, ntheta, endpoint=True)
s_plot = SurfaceRZFourier.from_wout(
    surface_filename, range="full torus",
    quadpoints_phi=quadpoints_phi, quadpoints_theta=quadpoints_theta
)

# optimize the currents in the TF coils and set up correct Bnormal
s, bs = coil_optimization(s_eq_preciseQH, bs, base_curves, curves, OUT_DIR, s_plot) 
bs.set_points(s.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

# check after-optimization average on-axis magnetic field strength
calculate_on_axis_B(bs, s)

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


#Initialize the permanent magnet class
pm_opt = PermanentMagnetGrid(
    s, rz_inner_surface=s_in, rz_outer_surface=s_out,
    Bn=Bnormal,
    #Nx=Nx,
    dr=dr,
    coordinate_flag=coordinate_flag,
)

pm_opt.geo_setup()

print('Number of available dipoles = ', pm_opt.ndipoles)

# Set some hyperparameters for the optimization
kwargs = initialize_default_kwargs('GPMO')
kwargs['K'] = 27000
kwargs['nhistory'] = 500

# Optimize the permanent magnets greedily
t1 = time.time()
R2_history, Bn_history, m_history = GPMO(pm_opt, algorithm = "baseline", **kwargs)
t2 = time.time()
print('GPMO took t = ', t2 - t1, ' s')

# plot the MSE history
iterations = np.linspace(0, kwargs['K'], len(R2_history), endpoint=False)
plt.figure()
plt.semilogy(iterations, R2_history, label=r'$f_B$')
plt.semilogy(iterations, Bn_history, label=r'$<|Bn|>$')
plt.grid(True)
plt.xlabel('Number of Magnets')
plt.ylabel('Metric values')
plt.legend()
plt.savefig(OUT_DIR + 'GPMO_MSE_history.png')

# Set final m to the minimum achieved during the optimization
min_ind = np.argmin(R2_history)
pm_opt.m = np.ravel(m_history[:, :, min_ind])

# Print effective permanent magnet volume
M_max = 1.465 / (4 * np.pi * 1e-7)
dipoles = pm_opt.m.reshape(pm_opt.ndipoles, 3)
print('Volume of permanent magnets is = ', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))) / M_max)
print('sum(|m_i|)', np.sum(np.sqrt(np.sum(dipoles ** 2, axis=-1))))

save_plots = True
if save_plots:
    # Save the MSE history and history of the m vectors
    #np.savetxt(OUT_DIR + 'mhistory_K' + str(kwargs['K']) + '_nphi' + str(nphi) + '_ntheta' + str(ntheta) + '.txt', m_history.reshape(pm_opt.ndipoles * 3, kwargs['nhistory'] + 1)) #this file occupies alot of space ~2gb, use with care
    np.savetxt(OUT_DIR + 'R2history_K' + str(kwargs['K']) + '_nphi' + str(nphi) + '_ntheta' + str(ntheta) + '.txt', R2_history)
    # Plot the SIMSOPT GPMO solution
    bs.set_points(s_plot.gamma().reshape((-1, 3)))
    Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
    make_Bnormal_plots(bs, s_plot, OUT_DIR, "biot_savart_optimized")

    # Look through the solutions as function of K and make plots
    for k in range(0, kwargs["nhistory"] + 1, 50):
        mk = m_history[:, :, k].reshape(pm_opt.ndipoles * 3)
        b_dipole = DipoleField(
            pm_opt.dipole_grid_xyz,
            mk,
            nfp=s.nfp,
            coordinate_flag=pm_opt.coordinate_flag,
            m_maxima=pm_opt.m_maxima,
        )
        b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
        b_dipole._toVTK(OUT_DIR + "Dipole_Fields_K" + str(int(kwargs['K'] / kwargs['nhistory'] * k)))
        print("Total fB = ",
              0.5 * np.sum((pm_opt.A_obj @ mk - pm_opt.b_obj) ** 2))
        Bnormal_dipoles = np.sum(b_dipole.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=-1)
        Bnormal_total = Bnormal + Bnormal_dipoles

        # For plotting Bn on the full torus surface at the end with just the dipole fields
        make_Bnormal_plots(b_dipole, s_plot, OUT_DIR, "only_m_optimized_K" + str(int(kwargs['K'] / kwargs['nhistory'] * k)))
        pointData = {"B_N": Bnormal_total[:, :, None]}
        s_plot.to_vtk(OUT_DIR + "m_optimized_K" + str(int(kwargs['K'] / kwargs['nhistory'] * k)), extra_data=pointData)

    # write solution to FAMUS-type file
    write_pm_optimizer_to_famus(OUT_DIR, pm_opt)

# Compute metrics with permanent magnet results
dipoles_m = pm_opt.m.reshape(pm_opt.ndipoles, 3)
num_nonzero = np.count_nonzero(np.sum(dipoles_m ** 2, axis=-1)) / pm_opt.ndipoles * 100
print("Number of possible dipoles = ", pm_opt.ndipoles)
print("% of dipoles that are nonzero = ", num_nonzero)

# Print optimized f_B and other metrics
### Note this will only agree with the optimization in the high-resolution
### limit where nphi ~ ntheta >= 64!
b_dipole = DipoleField(
    pm_opt.dipole_grid_xyz,
    pm_opt.m,
    nfp=s.nfp,
    coordinate_flag=pm_opt.coordinate_flag,
    m_maxima=pm_opt.m_maxima,
)
b_dipole.set_points(s_plot.gamma().reshape((-1, 3)))
bs.set_points(s_plot.gamma().reshape((-1, 3)))
Bnormal = np.sum(bs.B().reshape((qphi, ntheta, 3)) * s_plot.unitnormal(), axis=2)
f_B_sf = SquaredFlux(s_plot, b_dipole, -Bnormal).J()
print('f_B = ', f_B_sf)
B_max = 1.465
mu0 = 4 * np.pi * 1e-7
total_volume = np.sum(np.sqrt(np.sum(pm_opt.m.reshape(pm_opt.ndipoles, 3) ** 2, axis=-1))) * s.nfp * 2 * mu0 / B_max
print('Total volume = ', total_volume)

# Optionally make a QFM and pass it to VMEC
# This is worthless unless plasma
# surface is at least 64 x 64 resolution.
vmec_flag = False
if vmec_flag:
    from mpi4py import MPI
    from simsopt.mhd.vmec import Vmec
    from simsopt.util.mpi import MpiPartition
    mpi = MpiPartition(ngroups=1)
    comm = MPI.COMM_WORLD

    # Make the QFM surfaces
    t1 = time.time()
    Bfield = bs + b_dipole
    Bfield.set_points(s_plot.gamma().reshape((-1, 3)))
    qfm_surf = make_qfm(s_plot, Bfield)
    qfm_surf = qfm_surf.surface
    t2 = time.time()
    print("Making the QFM surface took ", t2 - t1, " s")

    # Run VMEC with new QFM surface
    t1 = time.time()

    ### Always use the QA VMEC file and just change the boundary
    vmec_input = "../../tests/test_files/input.LandremanPaul2021_QA"
    equil = Vmec(vmec_input, mpi)
    equil.boundary = qfm_surf
    equil.run()

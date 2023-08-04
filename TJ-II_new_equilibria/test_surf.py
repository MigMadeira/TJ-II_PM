import os
import glob
import shutil
import numpy as np
from pathlib import Path
from simsopt.util import MpiPartition
from simsopt.mhd import Vmec,  QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
mpi = MpiPartition()


vmec = Vmec('results/wout_ISTELL_nfp=1_final.nc', mpi=mpi, verbose=True, ntheta=80, nphi=80,range_surface="half period")
s = vmec.boundary
s.to_vtk("surf_nfp=1_final_test")

    
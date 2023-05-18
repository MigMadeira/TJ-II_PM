from pyevtk.hl import pointsToVTK 
import h5py
import numpy as np

filename = 'E:/MEFT/Tese/TJ2/input_tj2_wall.h5' #filename (the ".h5" file)
f = h5py.File(filename) #reads the h5 file

npoints = f['wall/wall_3D_0020994788/nelements'][0][0]
x1x2x3 = np.array(f['wall/wall_3D_0020994788/x1x2x3']).reshape((3,npoints))
y1y2y3 = np.array(f['wall/wall_3D_0020994788/y1y2y3']).reshape((3,npoints))
z1z2z3 = np.array(f['wall/wall_3D_0020994788/z1z2z3']).reshape((3,npoints))

#print(npoints)
#print(x1x2x3)
#print(y1y2y3)
#print(z1z2z3)

x = np.concatenate(x1x2x3)
y = np.concatenate(y1y2y3)
z = np.concatenate(z1z2z3)

pointsToVTK("vacuum_vessel", x, y, z)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 14:22:36 2022

@author: marcmath
"""

import seissolxdmf
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.colors as colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.linalg
from scipy.interpolate import griddata
from netCDF4 import Dataset
import scipy.ndimage as ndima
from scipy import signal
import pyproj

#

def project_3D_pts_on_plane(p_xyz,in_xyz):
# perpendicular projection of 3D points onto a plane
# xyzp = x,y,z coordinates of at least 3 pts belonging to the plane
# in_xyz = x,y,z coordinates of the points to project
# out_xyz = x,y,z coordinates of the projected points
    AB = p_xyz[1,:] - p_xyz[0,:]
    AC = p_xyz[2,:] - p_xyz[0,:]
    
    n=np.cross(AB,AC)
    d=np.dot(p_xyz[0,:],n.T)

    t0 = - (n[0]*in_xyz[:,0] + n[1]*in_xyz[:,1] + n[2]*in_xyz[:,2] - d)/(n[0]**2 + n[1]**2 + n[2]**2)
    X = in_xyz[:,0] + n[0]*t0;
    Y = in_xyz[:,1] + n[1]*t0;
    Z = in_xyz[:,2] + n[2]*t0;
    
    out_xyz = np.array([X, Y, Z]).T
    
    return out_xyz
        
def plane_coord(x1, x2, y1, y2, dip, strike, depth1, depth2):
    # depth 1: depth of the top bottom of the fault (= or > 0)
    # depth 2: en profondeur (< 0)
    # (x1,y1), (x2,y2) coordinates of the two top corners of the plane
    # out : xp,yp,zp  coordinates of the four corners of the plane
    
    dx = np.sin(np.deg2rad(strike-90)) * ( abs(depth2-depth1) / np.tan(np.deg2rad(dip)))
    dy = np.cos(np.deg2rad(strike-90)) * ( abs(depth2-depth1) / np.tan(np.deg2rad(dip)))

    xp = [x1, x2, x2 - dx, x1 - dx, x1]
    yp = [y1, y2, y2 + dy, y1 + dy, y1]
    zp = [depth1, depth1, depth2, depth2, depth1]
    
    return xp, yp, zp


def compute_unit_vec_trans_plane(xp,yp,zp):
    # xp,yp,zp coordinate of the 4 corners of the plane (but only 3 pts really needed)
    # hw, hh, th, tw: matrices and scalar needed to go from the flobal coordinates system
    # to the local coordinates system defined by the plane. 
    hw = np.array([xp[1], yp[1], zp[1]] - np.array([xp[2], yp[2], zp[2]])) # fault 3
    hw = hw / np.linalg.norm(hw)
    hh = np.array([xp[3], yp[3], zp[3]]) - np.array([xp[2], yp[2], zp[2]])
    hh = hh / np.linalg.norm(hh)
    th = -np.dot(np.array([xp[2], yp[2], zp[2]]), hh) 
    tw = -np.dot(np.array([xp[2], yp[2], zp[2]]), hw) 
    
    return hw, hh, th, tw 

def interpolate_2dgrid(xmin,xmax,ymin,ymax,step,data,method='cubic'):    
    # xmin,xmax,ymin,ymax,step : info needed to discretize the plane
    # data (3 x n matrix: x,y coordinates and value of the n data we want to interpolate)   
    l = xmax - xmin
    w = ymax - ymin
    
    nl = int(l / step)
    nw = int(w / step)
    
    xr = np.linspace(xmin, xmax, nl)
    yr = np.linspace(ymin, ymax, nw)
    
    data[:,0] = data[:,0] 
    data[:,1] = data[:,1]
     
    grid_x, grid_y = np.meshgrid(xr, yr)
    
    data_interp = griddata(data[:,0:2], data[:,2], (grid_x, grid_y), method='cubic')
    data_interp = np.nan_to_num(data_interp)
    
    return grid_x, grid_y, data_interp

def writeNetcdf4Paraview(sname, x, y, aName, aData):
    "create a netcdf file readable by paraview (but not by ASAGI)"
    fname = sname + "_paraview.nc"
    print("writing " + fname)
    ####Creating the netcdf file
    nx = x.shape[0]
    ny = y.shape[0]
    rootgrp = Dataset(fname, "w", format="NETCDF4")
    rootgrp.createDimension("u", nx)
    rootgrp.createDimension("v", ny)

    vx = rootgrp.createVariable("u", "f4", ("u",))
    vx[:] = x
    vy = rootgrp.createVariable("v", "f4", ("v",))
    vy[:] = y
    for i in range(len(aName)):
        vTd = rootgrp.createVariable(aName[i], "f4", ("v", "u"))
        vTd[:, :] = aData[i][:, :]
    rootgrp.close()


def writeNetcdf4SeisSol(sname, x, y, aName, aData):
    "create a netcdf file readable by ASAGI (but not by paraview)"
    ########## creating the file for SeisSol
    fname = sname + "_ASAGI.nc"
    print("writing " + fname)
    ####Creating the netcdf file
    nx = x.shape[0]
    ny = y.shape[0]

    rootgrp = Dataset(fname, "w", format="NETCDF4")

    rootgrp.createDimension("u", nx)
    rootgrp.createDimension("v", ny)

    vx = rootgrp.createVariable("u", "f4", ("u",))
    vx[:] = x
    vy = rootgrp.createVariable("v", "f4", ("v",))
    vy[:] = y
    ldata4 = [(name, "f4") for name in aName]
    ldata8 = [(name, "f8") for name in aName]
    mattype4 = np.dtype(ldata4)
    mattype8 = np.dtype(ldata8)
    mat_t = rootgrp.createCompoundType(mattype4, "material")

    # this transform the 4 D array into an array of tuples
    arr = np.stack([aData[i] for i in range(len(aName))], axis=2)
    newarr = arr.view(dtype=mattype8)
    newarr = newarr.reshape(newarr.shape[:-1])
    mat = rootgrp.createVariable("data", mat_t, ("v", "u"))
    mat[:] = newarr
    rootgrp.close()
    
############################################################################################
 # This script project the 3D slip model for the Maduo earthquake on a plane and
#build the 2D asagi file needed to use with FL33    
# The slip model is composed of three segments that are processed individually

projini = pyproj.Proj('EPSG:32647') # Projection of the slip model
projfinal = pyproj.Proj(proj='lcc', init='EPSG:3415') # projection of the mesh 


##################
### Segment 1 ###
#################
# Load the slip model
tri1 = np.loadtxt('../model/fault_geom_sgmt_1_tri.txt')
xyz1 = np.loadtxt('../model/fault_geom_sgmt_1_coord.txt')
slipmat1 = np.loadtxt('../model/fault_slip_sgmt_1.in')

# Convert slip model geometry in the same coordinates system as Maduo meshing
XY_0,XY_1 = pyproj.transform(projfinal,projini,xyz1[:,0],xyz1[:,1])
xyz1[:,0] = XY_0 
xyz1[:,1] = XY_1

XY_0,XY_1 = pyproj.transform(projfinal,projini,slipmat1[:,0],slipmat1[:,1])
slipmat1[:,0] = XY_0
slipmat1[:,1] = XY_1


bary1 = slipmat1[:,0:3]; # subfault barycenter
ss1   = slipmat1[:,3]; # strike-slip (m)
ds1   = slipmat1[:,4]; # dip-slip (m)
xy1    = xyz1[xyz1[:,2]==0]; #x,y coordinates of the surface fault trace
minx  = np.min(xy1[:,0])
maxx  = np.max(xy1[:,0])

# Compute the mean strike of the fault
p       = np.polyfit(xy1[:,0],xy1[:,1],1); 
ys1     = p[0] * np.array([minx, maxx]) + p[1];
alpha   = np.degrees(np.arctan((maxx-minx)/ (ys1[0]-ys1[-1])));
strike1 = 180 + alpha;

# Compute the coordinates of a plane that approximates the 3D fault
maxz1     = 0 # altitude max of the surface fault trace
depthmax1 = min(xyz1[:,2]) # km
dip1      = 81.6 #
xp1, yp1, zp1 = plane_coord(minx, maxx, ys1[0], ys1[-1], -dip1, strike1, maxz1, depthmax1)
p_xyz1 = np.array([xp1, yp1, zp1]).T # Coordinates of the plane corners

# compute scalar and vector needed to pass from 3D to 2D
ub1, ua1, ta1, tb1 = compute_unit_vec_trans_plane(xp1,yp1,zp1)

# project the barycenter of each subfault on the plane
bary_proj1 = project_3D_pts_on_plane(p_xyz1,bary1)

# Convert 3D coordinates in 2D coordinates
# Plane
xa  = np.dot(np.array([xp1, yp1, zp1]).T, ua1) + ta1
xb  = np.dot(np.array([xp1, yp1, zp1]).T, ub1) + tb1
xyp1 = np.transpose(np.vstack((xa, xb))) # coordinates of the plane corners in 2D

# subfault barycenter
xa  = np.dot(bary_proj1, ua1) + ta1
xb  = np.dot(bary_proj1, ub1) + tb1
xyb1 = np.transpose(np.vstack((xa, xb))) # coordinates of the subfaults barycenter in 2D

# Discretize the plane and interpolate the slip model
step1 = 800
grid_x1, grid_y1, ssi1 = interpolate_2dgrid(np.min(xyp1[:,0]),np.max(xyp1[:,0]),np.min(xyp1[:,1]),np.max(xyp1[:,1]),step1,np.c_[xyb1,ss1],'nearest')
grid_x1, grid_y1, dsi1 = interpolate_2dgrid(np.min(xyp1[:,0]),np.max(xyp1[:,0]),np.min(xyp1[:,1]),np.max(xyp1[:,1]),step1,np.c_[xyb1,ds1],'nearest')

ssi1[-1,:] = ssi1[-2,:]; # to avoid a line with 0 slip at the surface
dsi1[-1,:] = dsi1[-2,:]; # to avoid a line with 0 slip at the surface

# Smooth the slip model 
dsi1 = ndima.median_filter(dsi1,size=7)
ssi1 = ndima.median_filter(ssi1,size=7)

# Apply a tappering function to have a smooth decrease of slip at the edges of the fault
# Apply the tappering function along the lines
wind    = signal.windows.tukey(np.shape(dsi1)[1],alpha=0.25)
dsi1tuk = np.zeros(np.shape(dsi1))
ssi1tuk = np.zeros(np.shape(dsi1))
for i in range(0, np.shape(dsi1)[0]):
    dsi1tuk[i,:] = dsi1[i,:] * wind
    ssi1tuk[i,:] = ssi1[i,:] * wind
    
# Apply the tappering function along the rows
wind = signal.windows.tukey(np.shape(dsi1)[0],alpha=0.25)
wind[int(np.shape(dsi1)[0]/2)::] = 1 # do not tapper the slip at the surface
dsi1tuk2 = np.zeros(np.shape(dsi1))
ssi1tuk2 = np.zeros(np.shape(dsi1))

for i in range(0, np.shape(dsi1)[1]):
    dsi1tuk2[:,i] =  dsi1tuk[:,i] * wind   
    ssi1tuk2[:,i] =  ssi1tuk[:,i] * wind 

# Create the asagi file 
file_prefix = "fault1_ss_ds"
ldataName = ["strike-slip", "dip-slip"]
lgridded_myData = [ssi1tuk2, dsi1tuk2]
writeNetcdf4SeisSol(file_prefix, grid_x1[0,:], grid_y1[:,1], ldataName, lgridded_myData)
writeNetcdf4Paraview(file_prefix, grid_x1[0,:], grid_y1[:,1], ldataName, lgridded_myData)

##################
### Segment 2 ###
#################
# Load the slip model

tri2 = np.loadtxt('../model/fault_geom_sgmt_2_tri.txt')
xyz2 = np.loadtxt('../model/fault_geom_sgmt_2_coord.txt')
slipmat2 = np.loadtxt('../model/fault_slip_sgmt_2.in')

# Convert slip model geometry in the same coordinates system as Maduo meshing

XY_0,XY_1 = pyproj.transform(projfinal,projini,xyz2[:,0],xyz2[:,1])
xyz2[:,0] = XY_0
xyz2[:,1] = XY_1
XY_0,XY_1 = pyproj.transform(projfinal,projini,slipmat2[:,0],slipmat2[:,1])
slipmat2[:,0] = XY_0
slipmat2[:,1] = XY_1

bary2 = slipmat2[:,0:3]; # subfault barycenter
ss2 = slipmat2[:,3];  # strike-slip (m)
ds2 = slipmat2[:,4]; # dip-slip (m)
xy2=xyz2[xyz2[:,2]==0]; #x,y coordinates of the surface fault trace
minx2  = np.min(xy2[:,0])
maxx2  = np.max(xy2[:,0])

# Compute the mean strike of the fault
p      = np.polyfit(xy2[:,0],xy2[:,1],1); # fault 1
ys2     = p[0] * np.array([minx2, maxx2]) + p[1];
alpha  = np.degrees(np.arctan((maxx2-minx2)/ (ys2[0]-ys2[-1])));
strike2 = 180 + (alpha);


# Compute the coordinates of a plane that approximates the 3D fault
maxz2     = 0 # altitude max of the surface fault trace
depthmax2 = min(xyz2[:,2]) # km
dip2      = 81.6 #
xp2, yp2, zp2 = plane_coord(minx2, maxx2, ys2[0], ys2[-1], dip2, strike2, maxz2, depthmax2) 
p_xyz2 = np.array([xp2, yp2, zp2]).T #Coordinates of the plane corners

# compute scalar and vector needed to pass from 3D to 2D
ub2, ua2, ta2, tb2 = compute_unit_vec_trans_plane(xp2,yp2,zp2)

# project the barycenter of each subfault on the plane
bary_proj2 = project_3D_pts_on_plane(p_xyz2,bary2)

# Convert 3D coordinates in 2D coordinates
# Plane
xa  = np.dot(np.array([xp2, yp2, zp2]).T, ua2) + ta2
xb  = np.dot(np.array([xp2, yp2, zp2]).T, ub2) + tb2
xyp2 = np.transpose(np.vstack((xa, xb))) # coordinates of the plane corners in 2D

# subfault barycenter
xa  = np.dot(bary_proj2, ua2) + ta2
xb  = np.dot(bary_proj2, ub2) + tb2
xyb2 = np.transpose(np.vstack((xa, xb))) # coordinates of the subfaults barycenter in 2D

# Discretize the plane and interpolate the slip model
step2 = 800
grid_x2, grid_y2, ssi2 = interpolate_2dgrid(np.min(xyp2[:,0]),np.max(xyp2[:,0]),np.min(xyp2[:,1]),np.max(xyp2[:,1]),step2,np.c_[xyb2,ss2],'nearest')
grid_x2, grid_y2, dsi2 = interpolate_2dgrid(np.min(xyp2[:,0]),np.max(xyp2[:,0]),np.min(xyp2[:,1]),np.max(xyp2[:,1]),step2,np.c_[xyb2,ds2],'nearest')

# Smooth the slip model 
dsi2=ndima.median_filter(dsi2,size=7)
ssi2=ndima.median_filter(ssi2,size=7)

# Apply a tappering function to have a smooth decrease of slip at the edges of the fault
# Apply the tappering function along the lines
wind = signal.windows.tukey(np.shape(dsi2)[1],alpha=0.3)
dsi2tuk = np.zeros(np.shape(dsi2))
ssi2tuk = np.zeros(np.shape(dsi2))
for i in range(0, np.shape(dsi2)[0]):
    dsi2tuk[i,:] = dsi2[i,:] * wind
    ssi2tuk[i,:] = ssi2[i,:] * wind
    
# Apply the tappering function along the rows
wind = signal.windows.tukey(np.shape(dsi2)[0],alpha=0.3)
wind[int(np.shape(dsi2)[0]/2)::] = 1
dsi2tuk2 = np.zeros(np.shape(dsi2))
ssi2tuk2 = np.zeros(np.shape(dsi2))

for i in range(0, np.shape(dsi2)[1]):
    dsi2tuk2[:,i] =  dsi2tuk[:,i] * wind   
    ssi2tuk2[:,i] =  ssi2tuk[:,i] * wind  
    
    
# Create the asagi file 
file_prefix = "fault2_ss_ds"
ldataName = ["strike-slip", "dip-slip"]
lgridded_myData = [ssi2tuk2, dsi2tuk2]
writeNetcdf4SeisSol(file_prefix, grid_x2[0,:], grid_y2[:,1], ldataName, lgridded_myData)
writeNetcdf4Paraview(file_prefix, grid_x2[0,:], grid_y2[:,1], ldataName, lgridded_myData)

##################
### Segment 3 ###
#################
# Load the slip model
tri3 = np.loadtxt('../model/fault_geom_sgmt_3_tri.txt')
xyz3 = np.loadtxt('../model/fault_geom_sgmt_3_coord.txt')
slipmat3 = np.loadtxt('../model/fault_slip_sgmt_3.in')

# Convert slip model geometry in the same coordinates system as Maduo meshing
XY_0,XY_1 = pyproj.transform(projfinal,projini,xyz3[:,0],xyz3[:,1])
xyz3[:,0] = XY_0
xyz3[:,1] = XY_1
XY_0,XY_1 = pyproj.transform(projfinal,projini,slipmat3[:,0],slipmat3[:,1])
slipmat3[:,0] = XY_0
slipmat3[:,1] = XY_1

bary3 = slipmat3[:,0:3]; # subfault barycenter
ss3 = slipmat3[:,3];  # strike-slip (m)
ds3 = slipmat3[:,4]; # dip-slip (m)
xy3=xyz3[xyz3[:,2]==0]; #x,y coordinates of the surface fault trace
minx3  = np.min(xy3[:,0])
maxx3  = np.max(xy3[:,0])

# Compute the mean strike of the fault
p      = np.polyfit(xy3[:,0],xy3[:,1],1); 
ys3     = p[0] * np.array([minx3, maxx3]) + p[1];
alpha   = np.degrees(np.arctan((maxx3-minx3)/ (ys3[0]-ys3[-1])));
strike3 = 180 + (alpha);


# Compute the coordinates of a plane that approximates the 3D fault
maxz3     = 0# altitude max of the surface fault trace
depthmax3 = min(xyz3[:,2]) # km
dip3      = 81.6 # 
xp3, yp3, zp3 = plane_coord(minx3, maxx3, ys3[0], ys3[-1], -dip3, strike3, maxz3, depthmax3)
p_xyz3 = np.array([xp3, yp3, zp3]).T #Coordinates of the plane corners

# compute scalar and vector needed to pass from 3D to 2D
ub3, ua3, ta3, tb3 = compute_unit_vec_trans_plane(xp3,yp3,zp3)

# project the barycenter of each subfault on the plane
bary_proj3 = project_3D_pts_on_plane(p_xyz3,bary3)

# Convert 3D coordinates in 2D coordinates
# Plane
xa  = np.dot(np.array([xp3, yp3, zp3]).T, ua3) + ta3
xb  = np.dot(np.array([xp3, yp3, zp3]).T, ub3) + tb3
xyp3 = np.transpose(np.vstack((xa, xb))) # coordinates of the plane corners in 2D

# subfault barycenter
xa  = np.dot(bary_proj3, ua3) + ta3
xb  = np.dot(bary_proj3, ub3) + tb3
xyb3 = np.transpose(np.vstack((xa, xb))) # coordinates of the subfaults barycenter in 2D

# Discretize the plane and interpolate the slip model
step3 = 800
grid_x3, grid_y3, ssi3 = interpolate_2dgrid(np.min(xyp3[:,0]),np.max(xyp3[:,0]),np.min(xyp3[:,1]),np.max(xyp3[:,1]),step3,np.c_[xyb3,ss3],'nearest')
grid_x3, grid_y3, dsi3 = interpolate_2dgrid(np.min(xyp3[:,0]),np.max(xyp3[:,0]),np.min(xyp3[:,1]),np.max(xyp3[:,1]),step3,np.c_[xyb3,ds3],'nearest')

# Smooth the slip model 
dsi3=ndima.median_filter(dsi3,size=4)
ssi3=ndima.median_filter(ssi3,size=4)

# Apply a tappering function to have a smooth decrease of slip at the edges of the fault
# Apply the tappering function along the lines
wind = signal.windows.tukey(np.shape(dsi3)[1],alpha=0.3)
dsi3tuk = np.zeros(np.shape(dsi3))
ssi3tuk = np.zeros(np.shape(dsi3))
for i in range(0, np.shape(dsi3)[0]):
    dsi3tuk[i,:] = dsi3[i,:] * wind
    ssi3tuk[i,:] = ssi3[i,:] * wind
    
# Apply the tappering function along the rows
wind = signal.windows.tukey(np.shape(dsi3)[0],alpha=0.3)
wind[int(np.shape(dsi3)[0]/2)::] = 1
dsi3tuk2 = np.zeros(np.shape(dsi3))
ssi3tuk2 = np.zeros(np.shape(dsi3))

for i in range(0, np.shape(dsi3)[1]):
    dsi3tuk2[:,i] =  dsi3tuk[:,i] * wind   
    ssi3tuk2[:,i] =  ssi3tuk[:,i] * wind  
    
    
# Create the asagi file 
file_prefix = "fault3_ss_ds"
ldataName = ["strike-slip", "dip-slip"]
lgridded_myData = [ssi3tuk2, dsi3tuk2]
writeNetcdf4SeisSol(file_prefix, grid_x3[0,:], grid_y3[:,1], ldataName, lgridded_myData)
writeNetcdf4Paraview(file_prefix, grid_x3[0,:], grid_y3[:,1], ldataName, lgridded_myData)

################
### Figures ###
###############

plot_fault1 = True

if plot_fault1:
    # Segment 1
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot3D(bary_proj1[:,0], bary_proj1[:,1], bary_proj1[:,2], '.',label='projected subfault centroïd')
    ax.plot3D(xyz1[:,0], xyz1[:,1], xyz1[:,2], '.',label='subfault centroïd')
    ax.plot3D(xp1,yp1,zp1,'k',label='plane')
    ax.legend()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Segment 1 - Projected subfaults barycenter (m)')
    plt.show
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    collec = ax.plot_trisurf(xyz1[:,0], xyz1[:,1], xyz1[:,2], triangles=tri1-1, linewidth=0.2)
    collec.set_array(ss1)
    ax.view_init(10,95)
    cbar = fig.colorbar(collec)
    ax.set_box_aspect([6,1,1])  
    plt.xlabel('x (m)')
    plt.ylabel('y(m)')
    plt.title('Segment 1 - strike-slip (m)')
    

    fig = plt.figure()
    plt.pcolormesh(grid_x1, grid_y1, ssi1tuk2)
    plt.axis("equal")
    plt.show()
    plt.xlabel('Distance along strike (m)')
    plt.ylabel('Distance along width (m)')
    plt.title('Segment 1 - 2D slip model (m)')
    plt.colorbar()   
    
plot_fault2 = True

if plot_fault2:
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot3D(bary_proj2[:,0], bary_proj2[:,1], bary_proj2[:,2], '.',label='projected subfault centroïd')
    ax.plot3D(xyz2[:,0], xyz2[:,1], xyz2[:,2], '.',label='subfault centroïd')
    ax.plot3D(xp2,yp2,zp2,'k',label='plane')
    ax.legend()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Segment 2 - Projected subfaults barycenter (m)')
    plt.show
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    collec = ax.plot_trisurf(xyz2[:,0], xyz2[:,1], xyz2[:,2], triangles=tri2-1, linewidth=0.2)
    collec.set_array(ss2)
    ax.view_init(10,95)
    cbar = fig.colorbar(collec)
    ax.set_box_aspect([6,1,1])  
    plt.xlabel('x (m)')
    plt.ylabel('y(m)')
    plt.title('Segment 2 - strike-slip (m)')
    

    fig = plt.figure()
    plt.pcolormesh(grid_x2, grid_y2, ssi2tuk2)
    plt.axis("equal")
    plt.show()
    plt.xlabel('Distance along strike (m)')
    plt.ylabel('Distance along width (m)')
    plt.title('Segment 2 - 2D slip model (m)')
    plt.colorbar()   
    
plot_fault3 = True

if plot_fault3:
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot3D(bary_proj3[:,0], bary_proj3[:,1], bary_proj3[:,2], '.',label='projected subfault centroïd')
    ax.plot3D(xyz3[:,0], xyz3[:,1], xyz3[:,2], '.',label='subfault centroïd')
    ax.plot3D(xp3,yp3,zp3,'k',label='plane')
    ax.legend()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Segment 3 - Projected subfaults barycenter (m)')
    plt.show
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    collec = ax.plot_trisurf(xyz3[:,0], xyz3[:,1], xyz3[:,2], triangles=tri3-1, linewidth=0.2)
    collec.set_array(ss3)
    ax.view_init(10,95)
    cbar = fig.colorbar(collec)
    ax.set_box_aspect([3,1,1])  
    plt.xlabel('x (m)')
    plt.ylabel('y(m)')
    plt.title('Segment 3 - strike-slip (m)')
    

    fig = plt.figure()
    plt.pcolormesh(grid_x3, grid_y3, ssi3tuk2)
    plt.axis("equal")
    plt.show()
    plt.xlabel('Distance along strike (m)')
    plt.ylabel('Distance along width (m)')
    plt.title('Segment 3 - 2D slip model (m)')
    plt.colorbar()       

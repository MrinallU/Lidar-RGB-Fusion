import pykitti
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from open3d.j_visualizer import JVisualizer
# from open3d import JVisualizer

import os

# Load the dataset giving the folder, data and sample

basedir = os.getcwd()  # /home/sherlock/Documents/Kitti
date = '2011_09_26'
# Drive must coincide with the one listed in the folder
drive = '0005'

# Load the dataset
dataset = pykitti.raw(basedir, date, drive)
# Read the frame number 15 of the dataset
img = plt.imread("/home/mrinall/xviz/data/kitti/2011_09_26/2011_09_26_drive_0005_sync/"
                 "image_02/data/0000000098.png")
# Read the point cloud number 15 of the dataset
Velopoints = dataset.get_velo(98)
# Velopoints = client.XYZLut(metadata)(scan) # When applying the manual data

# Loading the calibration matrices
P_rect_20 = dataset.calib.P_rect_20  # Projection matrix  P_20 from Cam0 to Cam2 rectified
R_rect_20 = dataset.calib.R_rect_20  # Rotation matrix C_20 from Cam0 to Cam2 rectified
T_rect_20 = dataset.calib.T_cam0_velo_unrect  # Transformation matrix T_cam0Velo from Velo to Cam0

# Printing the camera calibration matrices
print('The rectified projection matrix P_rect_20 is: \n')
P_rect_20 = np.matrix(P_rect_20)
print(P_rect_20, P_rect_20.shape, type(P_rect_20), '\n')

print('The rectified rotation matrix R_rect_20 is: \n')
R_rect_20 = np.matrix(R_rect_20)
print(R_rect_20, R_rect_20.shape, type(R_rect_20))

print('The transformation matrix T_rect_20 is: \n')
T_rect_20 = np.matrix(T_rect_20)
print(T_rect_20, T_rect_20.shape, type(T_rect_20))

# Casting the 3D potins into float32
Velopoints = np.asarray(Velopoints, np.float32)

# Removing the reflectance component of the lidar measurement
Velopoints = Velopoints[:, :-1]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(Velopoints)

# visualizing the 3D point cloud
o3d.visualization.draw_geometries([pcd])
# visualizer = JVisualizer()
# visualizer.add_geometry(pcd)
# visualizer.show()

# Printing the first 10 Velopoints
# Each point is composed of X, Y, Z and Reflectance
print('The first 10 Velopoints are: \n\n', Velopoints[:10], Velopoints.shape, type(Velopoints))

# Projection matrix times Rotation matrix dim(3, 4)
T0 = P_rect_20 * R_rect_20
# Transformation matrix dim(3, 4)
T1 = T0 * T_rect_20
print('The Transformation matrix from the velodyne to camera 2 is: \n\n', T1)

# x: forward, y = left and z = up
# Find the indices of every point in the cloud that has value < 5
# Kitti authors suggest to use -5
idx = Velopoints[:, 0] < 5

# Delete the point clouds where x < 5
Velopoints = np.delete(Velopoints, np.where(idx), axis=0)

# Create a copy of the filtered Velopoints
Velopoints3P = np.copy(Velopoints)

# Save a txt file with the filtered points
np.savetxt('Raw_PointCloud_Total' + 'txt', Velopoints3P, delimiter=' ')

print('The Filtered points info is: \n\n', type(Velopoints3P), Velopoints3P.shape)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(Velopoints)

# visualizing the filtered 3D point cloud
o3d.visualization.draw_geometries([pcd])

# Number of Rows in the transformation matrix
dim_M = T1.shape[0]
# Number of Columns in the transformation matrix
dim_N = T1.shape[1]

print('Rows: ', dim_M, 'Columns: ', dim_N, '\n')

if Velopoints3P.shape[1] < dim_N:
    # Create a vector of ones with the same number of Velopoints
    ones_vect_Velo3p = np.ones((Velopoints3P.shape[0], 1), int)
    # Concatenate the Vector of ones into the Velopoints matrix as an additional column
    Velopoints3P = np.concatenate((Velopoints3P, ones_vect_Velo3p), axis=1)

# Cast the VeloPoints as a Matrix
Velopoints3P = np.matrix(Velopoints3P)
print('Velopoints3P or Velopoints3PP: \n\n', Velopoints3P[:10], Velopoints3P.shape, type(Velopoints3P))

# Copy the Velopoints
Velopoints3PP = np.copy(Velopoints3P)

# Transpose the point cloud to be multyplied by the transformation matrix
Velopoints3P = np.transpose(Velopoints3P)
print('Velopoints3P Transposed: ', Velopoints3P.shape, '\n')

# Transform the points from the Velodyne to Camera 2
y = T1 * Velopoints3P
# Transpose again the mapped points
y = np.transpose(y)

print('The mapped points from Velo to Cam 2 "y": \n\n', y[:10], y.shape, type(y), '\n')

# Extract the X and Y coordinate of the point cloud dim_M = 3
x_y = y[:, 0:dim_M - 1]

# Create a vector of ones of size (1, 2)
b_ones = np.ones((1, dim_M - 1), int)

# Extract the Z coordinates of the point cloud
z = y[:, dim_M - 1]

# Homogeneous coordinates of the camera points into image frame of camera 2
p_out = np.divide(x_y, np.multiply(z, b_ones))
print('X and Y coordinates in the image frame: \n\n', p_out[:10], p_out.shape, type(p_out), '\n')

# Obtain a list of indeces from 0 to the lenght of the VeloPoints in the image frame
idx_p_out = np.arange(start=0, stop=p_out.shape[0], step=1)
print('List of indeces: \n\n', idx_p_out, idx_p_out.shape, type(idx_p_out))

# Front image
plt.figure(figsize=(12, 4), dpi=100)
plt.imshow(img)
plt.show()

# X coordinate
x = p_out[:, 0]
# Y coordinate
y = p_out[:, 1]

plt.figure(figsize=(12, 4), dpi=100)
plt.plot(x, y, 'r.', markersize=0.5)

# Image Width
img_x = img.shape[1]
# Image Height
img_y = img.shape[0]

print('The Width of the image is:', img_x, 'The Height is:', img_y, '\n')

# Evaluation of the column x for identifying the values inside the image rank
# Find the X values inside the image's rank (list of true and false)
pointxt = np.logical_and(x >= 0, x <= img_x)
# Obtain the index of the values in the image rank; [0] are rows and [1] are columns.
idex = np.where(pointxt)[0]
# Obtain the points within the image bounds - Filtering only in X
# Recall that p_out are the X,Y velopoints in image frame
pout_ft = p_out[idex, :]

# Extract filtered indeces of point clouds in the image frame
idx_p_outt = idx_p_out[idex]
# X coordinate of the filtered points wrt to X
xa = pout_ft[:, 0]
# Y coordinate of the filtered points wrt to X
yb = pout_ft[:, 1]

# Evaluation of the column Y for identifying the values inside the image rank
# Find the Y values inside the image's rank (list of true and false)
pointyt = np.logical_and(yb >= 0, yb <= img_y)
# Obtain the index of the values in the image rank; [0] are rows and [1] are columns.
idexy = np.where(pointyt)[0]
# Obtain the points within the image bounds - Filtering wrt to X and now Y
pout_fty = pout_ft[idexy, :]

print('The filtered points wrt to X and Y "pout_fty" are: \n\n', pout_fty[:10], pout_fty.shape, '\n')
# Extract indeces of the point cloud coordinates in the image frame
idx_p_outtt = idx_p_outt[idexy]
print('The final indices "idx_p_outtt: after filtering are: \n\n', idx_p_outtt, idx_p_outtt.shape, '\n')

# X coordinate of the filtered points wrt to X and Y
xaa = pout_fty[:, 0]
# Y coordinate of the filtered points wrt to X and Y
ybb = pout_fty[:, 1]

# Rounding the final points and casting them to integers
pout_fty = np.matrix.round(pout_fty)
pout_fty = pout_fty.astype(int)
print('The filtered points "pout_fty_int" casted as integers are: \n\n', pout_fty[:10], pout_fty.shape, type(pout_fty))

# Plotting the points
fig = plt.figure(figsize=(18, 12))
ax1 = fig.add_subplot(211)
ax1.title.set_text('Points filtered w.r.t to X')
ax1.plot(xa, yb, 'r.', markersize=1)

ax2 = fig.add_subplot(212)
ax2.title.set_text('Points filtered w.r.t to X and Y')
ax2.plot(xaa, ybb, 'b.', markersize=1)

# Plotting the image + point cloud
plt.figure(figsize=(12, 4), dpi=100)
plt.imshow(img)
plt.plot(pout_fty[:, 0], pout_fty[:, 1], 'r.', markersize=1)

# Create an empty matrix with the shape of image size
matrix_points = np.zeros((img.shape[0], img.shape[1]))
print('The depth matrix shape is: ', matrix_points.shape)

for i in range(0, pout_fty.shape[0]):
    # Filtered X image coordinate
    x = pout_fty[i, 0]
    # Filtered Y image coordinate
    y = pout_fty[i, 1]

    # Extract the X, Y and Z coordinate of the point cloud in Velo coordinate frame
    ppoints = Velopoints3PP[idx_p_outtt[i], 0:3]
    # Compute the euclidian distance (Depth) of every point in the image frame
    matrix_points[y - 1, x - 1] = np.sqrt(np.power(ppoints[0], 2) + np.power(ppoints[1], 2) +
                                          np.power(ppoints[2], 2))

# np.savetxt('Data_set_pointcloud'+'.csv', matrix_points, delimiter=',')
print('Depth Matrix with points', matrix_points.shape)

# Configuring image
fig, ax = plt.subplots()
ax.imshow(img)
fig.set_size_inches(18, 12)

# Transforming X coordinates to list
c = pout_fty[:, 0].tolist()
# Transforming Y coordiantes to list
d = pout_fty[:, 1].tolist()
# X, Y and Z velopoints
f = Velopoints3PP[:, 0:3]
# Extracting filtered points of velodyne
f = f[idx_p_outtt]
# Computing the depth of the filtered given points
f = np.sqrt(np.power(f[:, 0], 2) + np.power(f[:, 1], 2) + np.power(f[:, 2], 2))
# Filtered points to list
f = f.tolist()

# Plotting
image_new = img
img = ax.scatter(c, d, s=3, marker=".", c=f, cmap=plt.cm.jet)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="1%", pad=0.05)
plt.colorbar(img, cax=cax)

plt.show()

imagePoints = Velopoints3PP[:, 0:3]
# Extracting filtered points of velodyne
imagePoints = imagePoints[idx_p_outtt]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(imagePoints)

# visualizing the 3D point cloud
o3d.visualization.draw_geometries([pcd])

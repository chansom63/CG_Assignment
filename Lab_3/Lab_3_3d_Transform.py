import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def translation_matrix(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ], dtype = float)

def scaling_matrix(sx, sy, sz):
    return np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ], dtype = float)

def rotation_z_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype = float)

#Applying the transformation

def apply_transform(points, M):
    hom = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed = (M @ hom.T).T
    return transformed[:, :3]

#Cube plotting
def plot_cube(ax, vertices, color = "blue", title = "cube"):
    for i, j in cube_edges:
        ax.plot(*zip(vertices[i], vertices[j]), color = color)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


#Cube definition
cube_vertices = np.array([
    [0,0,0],[1,0,0],[1,1,0],[0,1,0],
    [0,0,1],[1,0,1],[1,1,1],[0,1,1]
], float)

cube_edges = [(0,1),(1,2),(2,3),(3,0),
              (4,5),(5,6),(6,7),(7,4),
              (0,4),(1,5),(2,6),(3,7)]

S = scaling_matrix(1.5, 0.7, 1.2)
R = rotation_z_matrix(np.deg2rad(45))
T = translation_matrix(1, 0.5, -0.5)
M = T @ R @ S

transformed_vertices = apply_transform(cube_vertices, M)

fig = plt.figure(figsize = (10, 5))
ax1 = fig.add_subplot(121, projection = "3d")
plot_cube(ax1, cube_vertices, "blue", "original Cube")

ax2 = fig.add_subplot(122, projection = "3d")
plot_cube(ax2, transformed_vertices, "red", "Hybrid Transformed Cube")

plt.tight_layout()
plt.show()


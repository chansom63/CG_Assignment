import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon

# --------------------------
# Prism vertices
# --------------------------
prism_vertices = np.array([
    [0, 0, 0],   # v0
    [2, 0, 0],   # v1
    [1, 2, 0],   # v2
    [0, 0, 3],   # v3
    [2, 0, 3],   # v4
    [1, 2, 3]    # v5
])

# Edges
prism_edges = [
    (0,1), (1,2), (2,0),   # bottom triangle
    (3,4), (4,5), (5,3),   # top triangle
    (0,3), (1,4), (2,5)    # vertical edges
]

# --------------------------
# Projection functions
# --------------------------
def orthographic_projection(vertices):
    return vertices[:, :2]

def orthographic_projection_onto_plane(vertices, plane_normal, plane_point):
    projected = []
    for vertex in vertices:
        v = vertex - plane_point
        projection = vertex - np.dot(v, plane_normal) * plane_normal
        projected.append(projection)
    return np.array(projected)

def perspective_projection(vertices, proj_point):
    projected = []
    px, py, pz = proj_point
    for x, y, z in vertices:
        t = (pz) / (pz - z)
        x_proj = px + t * (x - px)
        y_proj = py + t * (y - py)
        projected.append([x_proj, y_proj])
    return np.array(projected)

def draw_edges(ax, projected_vertices, edges, color="b"):
    for edge in edges:
        p1, p2 = projected_vertices[edge[0]], projected_vertices[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color+"-o")

# --------------------------
# Perspective point (below base center)
# --------------------------
base_vertices = prism_vertices[:3]  # bottom triangle
base_center = np.mean(base_vertices, axis=0)
distance_below = 5
perspective_point = np.array([base_center[0], base_center[1], base_center[2] - distance_below])

# --------------------------
# Apply projections
# --------------------------
ortho_proj = orthographic_projection(prism_vertices)
persp_proj = perspective_projection(prism_vertices, perspective_point)

# --------------------------
# Rectangular face projection (V0,V1,V4,V3)
# --------------------------
rect_face_vertices = prism_vertices[[0, 1, 4, 3]]  # V0, V1, V4, V3
v1 = rect_face_vertices[1] - rect_face_vertices[0]
v2 = rect_face_vertices[3] - rect_face_vertices[0]
face_normal = np.cross(v1, v2)
face_normal = face_normal / np.linalg.norm(face_normal)
plane_point = np.mean(rect_face_vertices, axis=0)
rect_face_proj = orthographic_projection_onto_plane(prism_vertices, face_normal, plane_point)

x_axis = v1 / np.linalg.norm(v1)
y_axis = np.cross(face_normal, x_axis)

rect_face_2d = []
for point in rect_face_proj:
    vec = point - plane_point
    x = np.dot(vec, x_axis)
    y = np.dot(vec, y_axis)
    rect_face_2d.append([x, y])
rect_face_2d = np.array(rect_face_2d)

# --------------------------
# Orthographic projections of faces
# --------------------------
tri_faces = [[0,1,2], [3,4,5]]
rect_faces = [[0,1,4,3], [1,2,5,4], [2,0,3,5]]


# --------------------------
# 3D original prism with red edges and projection rays
# --------------------------
fig2 = plt.figure(figsize=(12,12))
ax3 = fig2.add_subplot(111, projection="3d")
ax3.set_title("Original Prism with Projection Rays")

# Draw prism edges in red
for edge in prism_edges:
    p1, p2 = prism_vertices[edge[0]], prism_vertices[edge[1]]
    ax3.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "r-", linewidth=2)

# Projection point
ax3.scatter(*perspective_point, color='k', s=100, label="Projection Point")

# Dashed rays to vertices
for vertex in prism_vertices:
    ax3.plot([perspective_point[0], vertex[0]],
             [perspective_point[1], vertex[1]],
             [perspective_point[2], vertex[2]],
             "k--", linewidth=1)

# Label corners with larger offset
offset3d = 0.5
for i, (x, y, z) in enumerate(prism_vertices):
    ax3.text(x+offset3d, y+offset3d, z+offset3d, f'V{i}', color="blue", fontsize=12, fontweight='bold',
             ha='center', va='center', backgroundcolor='w', alpha=0.7)

# Axis labels and view
ax3.set_xlabel("X", fontsize=14)
ax3.set_ylabel("Y", fontsize=14)
ax3.set_zlabel("Z", fontsize=14)
ax3.view_init(elev=20, azim=30)
plt.show()


# --------------------------
# Plot 2D projections
# --------------------------
fig, axes = plt.subplots(1, 3, figsize=(20,6))

label_offset_2d = -0.1  # offset for labels

# 1. Orthographic faces
axes[0].set_title("Orthographic Projection of Triangular Faces")
for tri in tri_faces:
    coords = np.vstack([ortho_proj[tri], ortho_proj[tri][0]])
    axes[0].plot(coords[:,0], coords[:,1], "g-o")
for rect in rect_faces:
    coords = np.vstack([ortho_proj[rect], ortho_proj[rect][0]])
    axes[0].plot(coords[:,0], coords[:,1], "m-o")
# Vertex labels with offset
for i, (x,y) in enumerate(ortho_proj):
    axes[0].text(x+label_offset_2d, y+label_offset_2d, f'V{i}', fontsize=12, fontweight='bold', color='blue',
                 ha='center', va='center', backgroundcolor='w', alpha=0.7)
axes[0].axis("equal"); axes[0].grid(True)

# 2. Rectangular face projection (moved to middle)
axes[1].set_title("Rectangular Face Projection (Plane Parallel to Face)")
rect_vertices = [0,1,4,3]
rect_coords = rect_face_2d[rect_vertices]
rect_patch = Polygon(rect_coords, alpha=0.3, color='cyan', edgecolor='blue', linewidth=3)
axes[1].add_patch(rect_patch)
rect_coords_closed = np.vstack([rect_coords, rect_coords[0]])
axes[1].plot(rect_coords_closed[:,0], rect_coords_closed[:,1], "b-o", linewidth=2, markersize=8)
# Label corners with offset
for i, vertex_idx in enumerate(rect_vertices):
    x, y = rect_coords[i]
    axes[1].text(x+label_offset_2d, y+label_offset_2d, f'V{vertex_idx}', fontsize=12, fontweight='bold', color="red",
                 ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
axes[1].axis("equal"); axes[1].grid(True)
axes[1].set_xlabel("X' (along face edge)"); axes[1].set_ylabel("Y' (perpendicular to face)")

# 3. Perspective projection (moved to last)
axes[2].set_title("Perspective Projection")
draw_edges(axes[2], persp_proj, prism_edges, "b")
for i in range(6):
    axes[2].text(persp_proj[i,0]+label_offset_2d, persp_proj[i,1]+label_offset_2d, f'V{i}', fontsize=12, fontweight='bold', color="purple",
                 ha='center', va='center', backgroundcolor='w', alpha=0.7)
axes[2].axis("equal"); axes[2].grid(True)

plt.tight_layout()
plt.show()

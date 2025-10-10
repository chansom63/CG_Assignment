import matplotlib.pyplot as plt
import numpy as np
import sys

# Increase recursion limit for fill algorithms, though an iterative approach is better for production.
sys.setrecursionlimit(10000)

# --- Configuration ---
GRID_SIZE = 101  # Grid dimensions (101x101)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# --- Helper Function to Draw Polygon Boundary ---

def draw_line(p1, p2, grid, color):
    """Draws a line on the grid using Bresenham's algorithm."""
    x1, y1 = p1
    x2, y2 = p2
    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx + dy

    while True:
        if 0 <= x1 < GRID_SIZE and 0 <= y1 < GRID_SIZE:
            grid[y1, x1] = color
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x1 += sx
        if e2 <= dx:
            err += dx
            y1 += sy

def draw_polygon_boundary(vertices, grid, color):
    """Draws the edges of a polygon on the grid."""
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)] # Wrap around to close the polygon
        draw_line(p1, p2, grid, color)

# --- 1. Scanline Fill Algorithm ---

def scanline_fill(vertices, grid, color):
    """Fills a polygon using the scanline algorithm."""
    min_y = min(v[1] for v in vertices)
    max_y = max(v[1] for v in vertices)

    for y in range(min_y, max_y + 1):
        intersections = []
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]

            # Check if the scanline at y intersects the edge (p1, p2)
            if p1[1] <= y < p2[1] or p2[1] <= y < p1[1]:
                # Calculate the x-intersection point
                if p1[1] != p2[1]: # Avoid horizontal lines
                    x_intersect = int(p1[0] + (y - p1[1]) / (p2[1] - p1[1]) * (p2[0] - p1[0]))
                    intersections.append(x_intersect)

        intersections.sort()

        # Fill between pairs of intersections
        for i in range(0, len(intersections), 2):
            if i + 1 < len(intersections):
                for x in range(intersections[i], intersections[i+1] + 1):
                    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                        grid[y, x] = color

# --- 2. Flood Fill Algorithm ---

def flood_fill(x, y, new_color, old_color, grid, connectivity=4):
    """
    Fills an area using an iterative flood fill (stack-based to avoid recursion depth issues).
    `connectivity` can be 4 or 8.
    """
    if (x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE or
            tuple(grid[y, x]) != old_color or tuple(grid[y, x]) == new_color):
        return

    stack = [(x, y)]

    while stack:
        px, py = stack.pop()

        if tuple(grid[py, px]) == old_color:
            grid[py, px] = new_color

            # Neighbors
            neighbors = []
            if connectivity == 4:
                neighbors = [(px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)]
            elif connectivity == 8:
                neighbors = [
                    (px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1),
                    (px + 1, py + 1), (px + 1, py - 1), (px - 1, py + 1), (px - 1, py - 1)
                ]

            for nx, ny in neighbors:
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    stack.append((nx, ny))

# --- 3. Boundary Fill Algorithm ---

def boundary_fill(x, y, fill_color, boundary_color, grid):
    """
    Fills an area using an iterative boundary fill (stack-based).
    Uses 4-way connectivity.
    """
    stack = [(x, y)]

    while stack:
        px, py = stack.pop()

        if (px < 0 or px >= GRID_SIZE or py < 0 or py >= GRID_SIZE):
            continue

        current_color = tuple(grid[py, px])
        if current_color != boundary_color and current_color != fill_color:
            grid[py, px] = fill_color
            stack.append((px + 1, py))
            stack.append((px - 1, py))
            stack.append((px, py + 1))
            stack.append((px, py - 1))


# --- Main Execution and Plotting ---

if __name__ == "__main__":
    # Define a triangle as the user-defined polygon
    polygon_vertices = [(20, 10), (80, 50), (30, 90)]

    # Calculate an interior point (centroid) for seed-based fills
    seed_point = (
        int(sum(v[0] for v in polygon_vertices) / 3),
        int(sum(v[1] for v in polygon_vertices) / 3)
    )

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Polygon Filling Algorithms', fontsize=16)

    # a) Scanline Fill
    grid_scanline = np.full((GRID_SIZE, GRID_SIZE, 3), WHITE, dtype=np.uint8)
    scanline_fill(polygon_vertices, grid_scanline, RED)
    draw_polygon_boundary(polygon_vertices, grid_scanline, BLACK) # Draw boundary for clarity
    axs[0, 0].imshow(grid_scanline, origin='lower')
    axs[0, 0].set_title('a) Scanline Fill')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    # b) Flood Fill (4-Connected)
    grid_flood4 = np.full((GRID_SIZE, GRID_SIZE, 3), WHITE, dtype=np.uint8)
    draw_polygon_boundary(polygon_vertices, grid_flood4, BLACK)
    flood_fill(seed_point[0], seed_point[1], RED, WHITE, grid_flood4, connectivity=4)
    axs[0, 1].imshow(grid_flood4, origin='lower')
    axs[0, 1].set_title('b) Flood Fill (4-Connected)')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    # b) Flood Fill (8-Connected)
    grid_flood8 = np.full((GRID_SIZE, GRID_SIZE, 3), WHITE, dtype=np.uint8)
    draw_polygon_boundary(polygon_vertices, grid_flood8, BLACK)
    flood_fill(seed_point[0], seed_point[1], RED, WHITE, grid_flood8, connectivity=8)
    axs[1, 0].imshow(grid_flood8, origin='lower')
    axs[1, 0].set_title('b) Flood Fill (8-Connected)')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])

    # c) Boundary Fill
    grid_boundary = np.full((GRID_SIZE, GRID_SIZE, 3), WHITE, dtype=np.uint8)
    draw_polygon_boundary(polygon_vertices, grid_boundary, BLACK)
    boundary_fill(seed_point[0], seed_point[1], RED, BLACK, grid_boundary)
    axs[1, 1].imshow(grid_boundary, origin='lower')
    axs[1, 1].set_title('c) Boundary Fill (4-Connected)')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
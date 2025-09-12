import matplotlib.pyplot as plt

# ---------------------------
# DDA Line Drawing Algorithm
# ---------------------------
def dda_line(x1, y1, x2, y2):
    points = []
    dx = x2 - x1
    dy = y2 - y1
    steps = int(max(abs(dx), abs(dy))) * 2   # finer steps for smoother line
    
    x_inc = dx / steps
    y_inc = dy / steps
    
    x, y = x1, y1
    for _ in range(steps + 1):
        points.append((x, y))   # keep floating values
        x += x_inc
        y += y_inc
    return points

# ---------------------------
# Bresenham’s Line Drawing Algorithm
# ---------------------------
def bresenham_line(x1, y1, x2, y2):
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return points

# ---------------------------
# Safe Input Function
# ---------------------------
def get_point(prompt):
    while True:
        try:
            x, y = map(int, input(prompt).split())
            return x, y
        except ValueError:
            print("Invalid input! Please enter two integers separated by space (e.g., 2 5).")

# ---------------------------
# Main Program
# ---------------------------
if __name__ == "__main__":
    print("Line Drawing using DDA and Bresenham's Algorithm")
    x1, y1 = get_point("Enter coordinates of the first point (x1 y1): ")
    x2, y2 = get_point("Enter coordinates of the second point (x2 y2): ")
    
    # Generate line points
    dda_points = dda_line(x1, y1, x2, y2)
    bresenham_points = bresenham_line(x1, y1, x2, y2)
    
    # Extract coordinates for plotting
    dda_x, dda_y = zip(*dda_points)
    bres_x, bres_y = zip(*bresenham_points)
    
    # Plot comparison
    plt.figure(figsize=(10, 5))
    
    # DDA Plot
    plt.subplot(1, 2, 1)
    plt.plot(dda_x, dda_y, "bo-", label="DDA Line")   # blue connected line
    plt.plot([x1, x2], [y1, y2], "r--", label="Ideal Line")
    plt.title("DDA Line Drawing (Smoothed)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    
    # Bresenham Plot
    plt.subplot(1, 2, 2)
    plt.plot(bres_x, bres_y, "go-", label="Bresenham Line")   # green connected line
    plt.plot([x1, x2], [y1, y2], "r--", label="Ideal Line")
    plt.title("Bresenham Line Drawing")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import time
import math

def plot_circle_points(xc, yc, x, y, points):
    """Helper function to add 8 symmetric points to a set to avoid duplicates."""
    points.add((xc + x, yc + y))
    points.add((xc - x, yc + y))
    points.add((xc + x, yc - y))
    points.add((xc - x, yc - y))
    points.add((xc + y, yc + x))
    points.add((xc - y, yc + x))
    points.add((xc + y, yc - x))
    points.add((xc - y, yc - x))

def midpoint_circle(xc, yc, r):
    """Implements the Mid-point Circle Drawing Algorithm."""
    points = set()
    x = 0
    y = r
    p = 1 - r

    plot_circle_points(xc, yc, x, y, points)

    while x < y:
        x += 1
        if p < 0:
            p += 2 * x + 1
        else:
            y -= 1
            p += 2 * (x - y) + 1
        plot_circle_points(xc, yc, x, y, points)

    return list(points)

def bresenham_circle_corrected(xc, yc, r):
    """Implements the CORRECTED Bresenham's Circle Drawing Algorithm."""
    points = set()
    x = 0
    y = r
    d = 3 - 2 * r

    plot_circle_points(xc, yc, x, y, points)

    while x < y:
        if d < 0:
            d = d + 4 * x + 6
        else:
            d = d + 4 * (x - y) + 10
            y -= 1
        x += 1
        plot_circle_points(xc, yc, x, y, points)

    return list(points)

def main():
    """Main function to get user input, run algorithms, and plot results."""
    try:
        xc = int(input("Enter the x-coordinate of the center (xc): "))
        yc = int(input("Enter the y-coordinate of the center (yc): "))
        r = int(input("Enter the radius of the circle (r): "))
        if r < 0:
            print("Radius cannot be negative.")
            return
    except ValueError:
        print("Invalid input. Please enter integer values.")
        return

    midpoint_points = midpoint_circle(xc, yc, r)
    bresenham_points = bresenham_circle_corrected(xc, yc, r)

    if not midpoint_points:
        print("No points were generated. Cannot plot. (This can happen if radius is 0).")
        return

    midpoint_points.sort(key=lambda p: math.atan2(p[1] - yc, p[0] - xc))
    bresenham_points.sort(key=lambda p: math.atan2(p[1] - yc, p[0] - xc))

    midpoint_points.append(midpoint_points[0])
    bresenham_points.append(bresenham_points[0])

    midpoint_x, midpoint_y = zip(*midpoint_points)
    bresenham_x, bresenham_y = zip(*bresenham_points)

    # --- Plotting ---
    plt.figure(figsize=(10, 10))

    plt.plot(midpoint_x, midpoint_y, color='blue', linewidth=2, label='Mid-point Algorithm')
    plt.plot(bresenham_x, bresenham_y, color='red', linestyle='--', linewidth=3, label="Bresenham's (Corrected)")

    # --- NEW: ADD TEXT LABEL DIRECTLY TO THE PLOT ---
    # Since both circles are identical, we only need one label.
    label_text = f"Circle at ({xc}, {yc}) with r={r}"
    # Position the text slightly above the top of the circle.
    text_x_position = xc
    text_y_position = yc + r + (r * 0.05) # Add a 5% margin above the circle

    plt.text(text_x_position, text_y_position, label_text,
             ha='center', va='bottom', fontsize=12, color='darkgreen')
    # --------------------------------------------------

    plt.title("Circle Drawing Algorithms")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend() # This displays the legend box for the lines
    plt.show()

if __name__ == "__main__":
    main()
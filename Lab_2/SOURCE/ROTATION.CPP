#include <graphics.h>
#include <conio.h>
#include <math.h>
#define PI 3.142

int x1 = 100, y1 = 100, x2 = 150, y2 = 50, x3 = 200, y3 = 100;

void rotate_triangle(float angle_deg, int color)
{
    float rad = angle_deg * PI / 180.0;

    // Find center of triangle
    float cx = (x1 + x2 + x3) / 3.0;
    float cy = (y1 + y2 + y3) / 3.0;

    // Rotate each point around center
    int xr1 = (x1 - cx) * cos(rad) - (y1 - cy) * sin(rad) + cx;
    int yr1 = (x1 - cx) * sin(rad) + (y1 - cy) * cos(rad) + cy;

    int xr2 = (x2 - cx) * cos(rad) - (y2 - cy) * sin(rad) + cx;
    int yr2 = (x2 - cx) * sin(rad) + (y2 - cy) * cos(rad) + cy;

    int xr3 = (x3 - cx) * cos(rad) - (y3 - cy) * sin(rad) + cx;
    int yr3 = (x3 - cx) * sin(rad) + (y3 - cy) * cos(rad) + cy;

    // Draw rotated triangle
    setcolor(color);
    line(xr1, yr1, xr2, yr2);
    line(xr2, yr2, xr3, yr3);
    line(xr3, yr3, xr1, yr1);
}

int main()
{
    int gd = DETECT, gm;
    initgraph(&gd, &gm, "F:\\BGI");

    setcolor(GREEN);
    line(x1, y1, x2, y2);
    line(x2, y2, x3, y3);
    line(x3, y3, x1, y1);

    getch();
    rotate_triangle(45, RED); // Rotate 45° about center
    getch();
    rotate_triangle(-45, BLUE); // Rotate -45° about center
    getch();

    closegraph();
    return 0;
}

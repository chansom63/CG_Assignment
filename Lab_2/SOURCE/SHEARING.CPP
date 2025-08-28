#include <graphics.h>
#include <conio.h>

int main()
{
    int gd = DETECT, gm;
    initgraph(&gd, &gm, "F:\\BGI");

    // Original triangle coordinates
    int x1 = 100, y1 = 100;
    int x2 = 150, y2 = 50;
    int x3 = 200, y3 = 100;

    // Draw original triangle
    setcolor(WHITE);
    line(x1, y1, x2, y2);
    line(x2, y2, x3, y3);
    line(x3, y3, x1, y1);

    getch();

    // Shearing factors
    float shx = 0.5; // X-direction shear
    float shy = 0.0; // Y-direction shear (0 for now)

    // Apply X-shear
    int sx1 = x1 + shx * y1;
    int sx2 = x2 + shx * y2;
    int sx3 = x3 + shx * y3;

    int sy1 = y1 + shy * x1;
    int sy2 = y2 + shy * x2;
    int sy3 = y3 + shy * x3;

    // Draw sheared triangle in RED
    setcolor(RED);
    line(sx1, sy1, sx2, sy2);
    line(sx2, sy2, sx3, sy3);
    line(sx3, sy3, sx1, sy1);

    getch();
    closegraph();
    return 0;
}

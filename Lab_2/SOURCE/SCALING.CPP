#include <graphics.h>
#include <conio.h>

int main()
{
    clrscr();
    int gd = DETECT, gm;
    initgraph(&gd, &gm, "F:\\BGI");
    setcolor(GREEN);
    line(100, 100, 400, 300);
    line(100, 100, 300, 400);
    line(300, 400, 400, 300);

    // Scaling factor values
    float Sx = 0.2;
    float Sy = 0.2;
    setcolor(RED);

    line(100 * Sx, 100 * Sy, 400 * Sx, 300 * Sy);
    line(100 * Sx, 100 * Sy, 300 * Sx, 400 * Sy);
    line(300 * Sx, 400 * Sy, 400 * Sx, 300 * Sy);
    getch();
    closegraph();
    return 0;
}
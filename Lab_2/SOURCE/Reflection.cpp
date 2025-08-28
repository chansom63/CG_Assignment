#include <graphics.h>
#include <conio.h>

int main()
{
    int gd = DETECT, gm;
    initgraph(&gd, &gm, "F:\\BGI");

    int midx = getmaxx() / 2;
    int midy = getmaxy() / 2;

    int x1 = midx - 50, y1 = midy - 50;
    int x2 = midx, y2 = midy - 100;
    int x3 = midx + 50, y3 = midy - 50;

    setcolor(WHITE);
    line(midx, 0, midx, getmaxx()); // y - axis
    line(0, midy, getmaxx(), midy); // x - axis

    setcolor(GREEN);

    line(x1, y1, x2, y2);
    line(x2, y2, x3, y3);
    line(x3, y3, x1, y1);

    getch();

    // Reflection across x - axis
    setcolor(RED);
    line(x1, 2 * midy - y1, x2, 2 * midy - y2);
    line(x2, 2 * midy - y2, x3, 2 * midy - y3);
    line(x3, 2 * midy - y3, x1, 2 * midy - y1);

    getch();
    closegraph();
    return 0;
}
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

    // Translation values
    int Tx = 200;
    int Ty = 10;
    setcolor(RED);

    line(100 + Tx, 100 + Ty, 400 + Tx, 300 + Ty);
    line(100 + Tx, 100 + Ty, 300 + Tx, 400 + Ty);
    line(300 + Tx, 400 + Ty, 400 + Tx, 300 + Ty);
    getch();
    closegraph();
    return 0;
}
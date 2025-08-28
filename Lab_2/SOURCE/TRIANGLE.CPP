#include <graphics.h>
#include <conio.h>

int main(){
	clrscr();
	int gd = DETECT, gm;
	initgraph(&gd, &gm, "F:\\BGI");
	setcolor(WHITE);
	line(100, 100, 400, 300);
	line(100, 100, 300, 400);
	line(300, 400, 400, 300);
	getch();
	closegraph();
	return 0;
}
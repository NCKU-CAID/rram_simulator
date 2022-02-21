# include <iostream>
# include <bits/stdc++.h>
# include <cstdio>

using namespace std;


int main(void) {
	freopen("test.grid.steady", "r", stdin);
	freopen("heatmap.txt", "w", stdout);
	int idx;
    float t;
    int x = 0;
    int y = 1;
	while (scanf("%d %f", &idx, &t) != EOF) {
        x = x + 1;
        if(x > 10 && y > 10){
            if(t > 350)
                cout << 8;
            else if(t > 345)
                cout << 7;
            else if(t > 340)
                cout << 6;
            else if(t > 335)
                cout << 5;
            else if(t > 330)
                cout << 4;
            else if(t > 325)
                cout << 3;
            else if(t > 320)
                cout << 2;
            else if(t > 315)
                cout << 1;
            else if(t > 310)
                cout << 0;
        }
        if(x == 64){
            x = 0;
            y = y + 1;
            if(y > 11)
                cout << endl;
        }
	}
	return 0;
}

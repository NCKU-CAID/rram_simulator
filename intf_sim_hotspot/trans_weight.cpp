# include <iostream>
# include <bits/stdc++.h>
# include <cstdio>

using namespace std;


int main(void) {
	freopen("../weight_file.txt", "r", stdin);
	freopen("weight_sum.txt", "w", stdout);
	int t;
	int count = 0;
	double sum = 0.0;
	while (scanf("%d", &t) != EOF) {
		count ++;
		if(t != 0)
			sum += 1.21 / (double)t;
		if (count == 114688) {
			count = 0;
			cout << sum << endl;
			sum = 0;
		}
	}
	cout << sum << endl;
	return 0;
}

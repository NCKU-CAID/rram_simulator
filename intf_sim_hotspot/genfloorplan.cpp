# include <iostream>
# include <cstdio>

using namespace std;


int main(void) { 
    freopen("test.flp", "w", stdout);
    float a = 0.007012, b = 0.007312, c = 0.007612;
    float a2 = 0, b2 = 0.00028, c2 = 0.00039, d2 = 0.000587, e2 = 0.00058, f2 = 0.000503, g2 = 0.0;
    int cnt = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    int l = 0;
    
    cout << "DRAM_0"                 	             << "\t"  << "0.007012" << "\t" << "0.018000" << "\t"  << fixed << "0.000000" << "\t" << "0.000000" << endl;
    cout << "subarray_00"                            << "\t"  << "0.009988" << "\t" << "0.000136" << "\t"  << fixed << "0.007012" << "\t" << "0.017864" << endl;

    for (i = 0; i < 116; i++) {
	cout << "eDRAM_buffer_"			<< i << "\t"  << "0.000300" << "\t" << "0.000280" << "\t"  << fixed << a << "\t" << a2 + 0.000616*cnt << endl;
	cout << "eDRAM_to_IMA_bus_"		<< i << "\t"  << "0.000300" << "\t" << "0.000300" << "\t"  << fixed << a << "\t" << b2 + 0.000616*cnt << endl;
	cout << "Router_"			<< i << "\t"  << "0.000300" << "\t" << "0.000113" << "\t"  << fixed << b << "\t" << c2 + 0.000616*cnt << endl;
	cout << "Sigmoid_SA_Maxpool_OR_"	<< i << "\t"  << "0.000300" << "\t" << "0.000029" << "\t"  << fixed << a << "\t" << d2 + 0.000616*cnt << endl;
	cout << "ADC_"        			<< i << "\t"  << "0.000300" << "\t" << "0.000390" << "\t"  << fixed << b << "\t" << a2 + 0.000616*cnt << endl;
	cout << "DAC_"          		<< i << "\t"  << "0.000300" << "\t" << "0.000007" << "\t"  << fixed << a << "\t" << e2 + 0.000616*cnt << endl;
	cout << "SH_SA_IR_OR_"  		<< i << "\t"  << "0.000300" << "\t" << "0.000113" << "\t"  << fixed << b << "\t" << f2 + 0.000616*cnt << endl;
	l = 0;
	k = 0;
	for (j = 0; j < 14; j++){
		cout << "subarray_" << i * 14 + j << "\t\t"  << "0.000044" << "\t" << "0.000308" << "\t"  << fixed << c + 0.000044 * l << "\t" << g2 + 0.000308*k  + 0.000616*cnt<< endl;
		k++;
		if(k == 2){
			l++;
			k = 0;
		}
	}
	cout << endl << endl;
        if (cnt == 28) {
            cnt = 0;
            a += 0.000908;
            b += 0.000908;
	    c += 0.000908;
        }
        else
            cnt++;
    }
    a  = 0.010644;
    a2 = 0.000000;
    cnt = 0;
    for(i = 116; i < 319; i++){
    	cout << "tile_"                 << i << "\t"  << "0.000908" << "\t" << "0.000616" << "\t"  << fixed << a << "\t" << a2 + 0.000616*cnt << endl;
	if(cnt == 28){
	    cnt = 0;
	    a += 0.000908;
	}
	else
	    cnt++;
    }
    return 0;
}

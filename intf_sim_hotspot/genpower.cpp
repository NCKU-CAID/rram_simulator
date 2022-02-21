#include <iostream>
#include <cstdio>
using namespace std;

float arr[1623]={0};


int main(int argc, char *argv[]){
    freopen("weight_sum.txt", "r", stdin);
    freopen("test.ptrace", "w", stdout);

    int i = 0, j = 0;
    float tmp;

	for(i=0; i< 1623; i++)
		cin >> arr[i];
	
	cout << "DRAM_0"         << "\t";
	cout << "subarray_00"    << "\t";
	
	for (i = 0; i < 116; i++) {
		cout << "eDRAM_buffer_"		<< i << "\t"
		     << "eDRAM_to_IMA_bus_"	<< i << "\t"
		     << "Router_"		<< i << "\t"
		     << "Sigmoid_SA_Maxpool_OR_"<< i << "\t"
		     << "ADC_"        		<< i << "\t"
		     << "DAC_"          	<< i << "\t"
		     << "SH_SA_IR_OR_" 		<< i << "\t";    
		for (j = 0; j < 14; j++){
			cout << "subarray_"	<< i * 14 + j << "\t"; 
		}
	}
	for (i = 116; i < 319; i++){
		cout << "tile_"         << i << "\t";
	}
	for (int k=0; k <2; k++){
	    cout << endl;
	    cout << 0.4083  << fixed << "\t";
	    cout << 0.0001  << fixed << "\t";
	    for (i = 0; i < 116; i++) {
	    	cout << 0.0207  << fixed << "\t" \
		     << 0.007   << fixed << "\t" \
		     << 0.0105  << fixed << "\t" \
		     << 0.00265 << fixed << "\t" \
		     << 0.192   << fixed << "\t" \
		     << 0.048   << fixed << "\t" \
		     << 0.02016 << fixed << "\t";
		for (j = 0; j < 14; j++){
			// cin >> arr[i*96+j];
			cout << arr[i*14+j]  << "\t";
		}
	    }
	    for (i = 116; i < 319; i++){
	    	cout << 0.30101  << fixed << "\t";
	    }
	}
    return 0;
}

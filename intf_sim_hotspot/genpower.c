#include "stdio.h"
#include "string.h"
#include "stdlib.h"

#define LOCAL_ITEM_NUM    7
#define MAX_CHAR    100

float weight[1623]={0};

struct item {
    char name[MAX_CHAR];
    float value;
};

struct item itemlist[LOCAL_ITEM_NUM];

void read_cfg (char *fname) {
    FILE *ptr = fopen(fname, "r");
    if (ptr == NULL) {
        printf("Open file \"%s\" error !\n", fname);
        exit(1);
    }
    int cnt = 0;
    char buffer[MAX_CHAR];
    char item_name[MAX_CHAR];
    float item_val;

    while (fgets(buffer, MAX_CHAR, ptr) != NULL) {
        if (buffer[0] == '#' | strlen(buffer) < 3) {
            continue;
        }
        sscanf(buffer, "%s %f", item_name, &item_val);
        strncpy(itemlist[cnt].name, item_name, MAX_CHAR);
        itemlist[cnt].value = item_val;
        cnt ++;
    }
    //printf ("READ DONE, VALID LINE : %d\n", cnt);
    fclose(ptr);
}

int main(int argc, char *argv[]){
    FILE *r_fptr, *w_fptr;
    // I/O file checking
    if (argc < 4) {
        printf("\nLack of files !\n");
        printf("need 3 files, circuit power consumption, subarray weight file and output file\n");
        printf("Parameters sequence : circuit_power.cfg/weight.cfg/consumption.ptrace\n\n");
        exit(1);
    } else {
        if ((r_fptr = fopen(argv[2] ,"r")) == NULL) {
            printf("Wrror opening weight file !\n");
            exit(1);
        }
        if ((w_fptr = fopen(argv[3], "w")) == NULL){
            printf("Error creating output file !\n");
            exit(1);
        }
    }
    // read circuit power configuration
    read_cfg(argv[1]);

    int i = 0, j = 0, k = 0;
    float tmp;

	for(i = 0; i < 1623; i++) {
       fscanf(r_fptr, "%f", &weight[i]);
    }
    fclose(r_fptr);
	
    fprintf(w_fptr, "DRAM_0\t" );
    fprintf(w_fptr, "subarray_00\t");
	
	for (i = 0; i < 116; i++) {
        //fprintf(w_fptr, "eDRAM_buffer_%d\t"              , i);
        //fprintf(w_fptr, "eDRAM_to_IMA_bus_%d\t"          , i);
        //fprintf(w_fptr, "Router_%d\t"                    , i);
        //fprintf(w_fptr, "Sigmoid_SA_Maxpool_OR_%d\t"     , i);
        //fprintf(w_fptr, "ADC_%d\t"                       , i);
        //fprintf(w_fptr, "DAC_%d\t"                       , i);
        //fprintf(w_fptr, "SH_SA_IR_OR_%d\t"               , i);
        for (k = 0; k < LOCAL_ITEM_NUM; k++) {
            fprintf(w_fptr, "%s_%d\t", itemlist[k].name, i);
        }
		for (j = 0; j < 14; j++){
            fprintf(w_fptr, "subarray_%d\t", i * 14 + j);
		}
	}
	for (i = 116; i < 319; i++){
        fprintf(w_fptr, "tile_%d\t", i);
	}
	for (k=0; k <2; k++){
        fprintf(w_fptr, "\n");
        fprintf(w_fptr, "0.4083    \t");
        fprintf(w_fptr, "0.0001    \t");
	    for (i = 0; i < 116; i++) {
            //fprintf(w_fptr, "0.0207    \t0.007    \t0.0105   \t0.00265  \t0.192     \t0.048    \t0.02016  \t");
            for (int idx = 0; idx < LOCAL_ITEM_NUM; idx++){
                fprintf(w_fptr, "%7f\t", itemlist[idx].value);
            }
		    for (j = 0; j < 14; j++){
                fprintf(w_fptr, "%7f\t", weight[i * 14 + j] );
		    }
	    }
	    for (i = 116; i < 319; i++){
            fprintf(w_fptr, "0.30101  \t");
	    }
	}

    fclose(w_fptr);

    return 0;
}

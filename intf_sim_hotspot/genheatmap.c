#include "stdio.h"
#include "stdlib.h"

int main(int argc, char *argv[]) {
    
    FILE *r_fptr, *w_fptr;

    r_fptr = fopen(argv[1], "r");

    int buffer1, buffer2;
    float buffer3;

    int temperature[8];
    int level[8];

    int i, j;

    i = 0;
    while(fscanf(r_fptr, "%d %d", &buffer1, &buffer2) != EOF){
        temperature[i] = buffer1;
        level[i] = buffer2;
        i++;
    }

    fclose(r_fptr);

    r_fptr = fopen(argv[2], "r");
    w_fptr = fopen(argv[3], "w");

    i = 0;
    j = 1;
    while(fscanf(r_fptr, "%d %f", &buffer1, &buffer3) != EOF){
        i++;
        if(i > 10 && j > 10){
            if(buffer3 > temperature[0])
                fprintf(w_fptr, "%d", level[0]);
            else if(buffer3 > temperature[1])
                fprintf(w_fptr, "%d", level[1]);
            else if(buffer3 > temperature[2])
                fprintf(w_fptr, "%d", level[2]);
            else if(buffer3 > temperature[3])
                fprintf(w_fptr, "%d", level[3]);
            else if(buffer3 > temperature[4])
                fprintf(w_fptr, "%d", level[4]);
            else if(buffer3 > temperature[5])
                fprintf(w_fptr, "%d", level[5]);
            else if(buffer3 > temperature[6])
                fprintf(w_fptr, "%d", level[6]);
            else
                fprintf(w_fptr, "%d", level[7]);
        }
        if(i == 64){
            i = 0;
            j++;
            if(j > 11)
                fprintf(w_fptr, "\n");
        }
    }

    fclose(w_fptr);
    fclose(r_fptr);

	return 0;
}

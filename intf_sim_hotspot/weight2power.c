# include "stdio.h"
# include "string.h"
# include "stdlib.h"

# define MAX_CHAR 100
# define ITEM_NUM 10

float voltage;
int subarray_num;

void read_cfg(char *fname)
{
    printf("reading subarray number and voltage...\n");
    FILE *fptr = fopen(fname, "r");
    //if (fptr == NULL) {
    //    printf("Open file \"%s\" error !\n", fname);
    //    exit(1);
    //}
    //int subarray_num;
    //float voltage;
    char buffer[MAX_CHAR];
    char temp[MAX_CHAR];
    fgets(buffer, MAX_CHAR, fptr);
    sscanf(buffer, "%s %f", temp, &voltage);

    fgets(buffer, MAX_CHAR, fptr);
    sscanf(buffer, "%s %d", temp, &subarray_num);
    
    fclose(fptr);
    printf("Voltage : %f\n", voltage);
    printf("Subarray number : %d\n", subarray_num);
}

void check_args(int argc, char **argv) 
{
    printf("Check arguments...\n");
    FILE *weight_file, *cfg_file;
    FILE *write_file;
    if (argc < 4) {
        printf("Lack of file...\n");
        printf("Parameters sequence : weight_file.txt   weight2power.cfg  output_file\n");
        exit(1);
    } else {
        if ((weight_file = fopen(argv[1], "r")) == NULL) {
            printf("Error opening weight file...\n");
            exit(1);
        }
        if ((cfg_file = fopen(argv[2], "r")) == NULL ) {
            printf("Error opening configuration file...\n");
            exit(1);
        }
        if ((write_file = fopen(argv[3], "w")) == NULL) {
            printf("Error creating write file...\n");
            exit(1);
        }
    }
    fclose(weight_file);
    fclose(cfg_file);
    fclose(write_file);
}

// read weight and calculate and write output
void calculate(char *weight_file, char *output_file)
{
    printf("Reading weight file...\n");
    FILE *weightptr = fopen(weight_file, "r");
    FILE *writeptr = fopen(output_file, "w");
    int cnt = 0; 
    float in;
    double sum = 0;
    // read weight file
    float vol_square = voltage * voltage;
    printf("Reading input...\n");
    while(fscanf(weightptr, "%f", &in) != EOF) {
        cnt += 1;
        if (in != 0) {
            sum += (vol_square/(double)(in));
        }
        if (cnt == subarray_num) {
            
            fprintf(writeptr, "%f\n", sum);
            cnt = 0;
            sum = 0.0;
        }
    }
    fprintf(writeptr, "%f\n", sum);
    printf("Generate output file %s\n", output_file);

    fclose(weightptr);
    fclose(writeptr);
}

int main(int argc, char *argv[])
{
    // FILE *weight_file, *cfg_file;
    // FILE *write_file;
    check_args(argc, argv);

    
    // weight_file = fopen(argv[1], "r");
    // cfg_file = fopen(argv[2], "r");
    // write_file = fopen(argv[3], "w");

    read_cfg(argv[2]);
    calculate(argv[1], argv[3]);

    return 0;
}

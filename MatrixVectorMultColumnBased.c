#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define MASTER 0

float *create1DArray(int n) {
    float *T = (float *)malloc(n * sizeof(float));
    return T;
}

void fillArray(float *T, int n) {
    int i;
    for (i = 0; i < n; i++)
        T[i] = (float)i + 1.0;
}
void fillArrayZero(float *T, int n) {
    int i;
    for (i = 0; i < n; i++)
        T[i] = 0.0;
}
void printArray(float *T, int n) {paralel codes
    int i;
    for (i = 0; i < n; i++)
        printf("%.2f ", T[i]);
    puts("");
}
float *multp(float *A_local, float *v_local, int n1, int chunk){
  float *final = create1DArray(n1);
  fillArrayZero(final,n1);
  int i,counter=0;
  for (i = 0; i < n1* chunk; i++) {
    final[i%n1] += A_local[i] * v_local[counter];
    //printf("%f\t %d \n",final[i%n1],counter);
    //Doing necessary mult. with basic algorithm
    if(i+1==n1){
      counter++;
    }
    if(counter == chunk){
      break; //if i reach out of border of vector im breaking the loop
    }
  }
  return final;
}
int main(int argc, char *argv[]) {

int n1 = atoi(argv[1]);
int n2 = atoi(argv[2]);

int rank, size, i;

MPI_Init(NULL, NULL);

MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

MPI_Status stat;

float *A, *A_local, *Vector, *v_local, *total, *last;
double tcomm1, tcomm2, tcal1, tcal2,t1,t2;
t1 = MPI_Wtime();
int chunk = n2 / size;

A_local = create1DArray(chunk*n1);//A(Main matris), V (Vector)
v_local = create1DArray(chunk); //created local var.s
total = create1DArray(n1);


if (rank == MASTER) {
    A = create1DArray(n1 * n2);
    fillArray(A, n1 * n2);
    Vector = create1DArray(n2);// Main var.s created in the master
    fillArray(Vector, n2);
    last = create1DArray(n1);
}

MPI_Datatype colType, newColType;//doing transaction with new dataTypes

int blocklength = 1, stride = n2, count = n1;
MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &colType);
MPI_Type_commit(&colType);
MPI_Type_create_resized(colType, 0, 1*sizeof(float), &newColType);
MPI_Type_commit(&newColType);
tcomm1=MPI_Wtime();
MPI_Scatter(Vector, chunk, MPI_FLOAT, v_local, chunk, MPI_FLOAT,MASTER, MPI_COMM_WORLD);
// Collective communication using colType
MPI_Scatter(A, chunk, newColType, A_local, chunk*n1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
tcal1=MPI_Wtime();
total = multp(A_local, v_local, n1, chunk);
tcal2=MPI_Wtime();
MPI_Barrier(MPI_COMM_WORLD);
MPI_Reduce(total,last,n1,MPI_FLOAT,MPI_SUM,MASTER,MPI_COMM_WORLD);
tcomm2=MPI_Wtime();
t2 = MPI_Wtime();
if (rank == MASTER){
    //printArray(A_local, chunk*n1);
    //printArray(v_local, chunk);
    printf("Elapsed time = %f sec.\n", t2 - t1);
    printf("Elapsed time for comm = %f sec.\n", tcomm2 - tcomm1);
  }
  printf("Rank = %d\tElapsed time for calculation = %f sec.\n", rank,tcal2 - tcal1);

MPI_Finalize();

}

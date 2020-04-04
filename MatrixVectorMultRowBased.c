#include <stdio.h>
#include <stdlib.h>
#include "mpi.h" //MPI library
#include <math.h>

#define MASTER 0

float *create1DArray(float n) {
     float *T = (float *)malloc(n * sizeof(float));
     return T;
}
void fillArray(float *T, int n){
  int i;
  for(i=0;i<n;i++){
    T[i] = 1.0 + i;
  }
}
void printArray(float *T, int n) {
     int i;
     for (i = 0; i < n; i++)
          printf("%.2f ", T[i]);

     puts("");
}
float innerProduct(float *u, float *v, int n){
  int i;
  float sum=0.0;

  for (i = 0; i < n; i++) {
    sum += u[i] * v[i];
  }
  return sum;
}
float *mat_vec_mult(float *M, float *v, int n1, int n2){
  int i;
  float *r =  create1DArray(n1);
  for(i =0; i< n1;i++)
    r[i] = innerProduct(&M[i*n2], v, n2);
  return r;
}

int main(int argc, char const *argv[]) {
  int n1 = atoi(argv[1]);
  int n2 = atoi(argv[2]);
  int rank, size, i;
  MPI_Init(NULL,NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int chunk = n1 / size;

  float *A, *x, *b, *A_local, *B_local;
  double tcomm1, tcomm2, tcal1, tcal2,t1,t2;
  t1 = MPI_Wtime();
  x = create1DArray(n2);
  A_local = create1DArray(chunk * n2);
  B_local = create1DArray(chunk);

  if(rank ==MASTER){
    A = create1DArray(n1 * n2);
    fillArray(A, n1 * n2);
    fillArray(x, n2);
    b = create1DArray(n1);
  }
  MPI_Datatype rowType;
  MPI_Type_contiguous(n2, MPI_FLOAT, &rowType);
  MPI_Type_commit(&rowType);
  tcomm1 = MPI_Wtime();
  MPI_Bcast(x, n2 , MPI_FLOAT, MASTER, MPI_COMM_WORLD);
  MPI_Scatter(A, chunk , rowType, A_local, chunk , rowType, MASTER, MPI_COMM_WORLD);
  tcomm2 = MPI_Wtime();
  tcal1 = MPI_Wtime();
  B_local = mat_vec_mult(A_local, x, chunk, n2); // local comp
  tcal2 = MPI_Wtime();
  MPI_Gather(B_local, chunk, MPI_FLOAT, b, chunk, MPI_FLOAT, MASTER, MPI_COMM_WORLD);

  t2 = MPI_Wtime();
  if (rank==MASTER) {
    //printArray(b, n1);
    printf("Elapsed time = %f sec.\n", t2 - t1);
    printf("Elapsed time for comm = %f sec.\n", tcomm2 - tcomm1);
  }
  printf("Rank = %d\tElapsed time for calculation = %f sec.\n", rank,tcal2 - tcal1);

  MPI_Finalize();
}

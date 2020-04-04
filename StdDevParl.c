#include <stdio.h>
#include <stdlib.h>
#include "mpi.h" //MPI Library
#include <math.h>

#define MASTER 0
#define N 3000000

float *create1DArray(float n) {
     float *T = (float *)malloc(n * sizeof(float));
     return T;
}

void printArray(float *T, int n) {
     int i;
     for (i = 0; i < n; i++)
          printf("%f ", T[i]);

     puts("");
}

int main(void) {

int rank, size, i;
MPI_Init(NULL, NULL);

MPI_Status status;
double t1, t2;
t1 = MPI_Wtime(); // start clock

MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

int chunk = N / size;

float *array;  // each process has its own array address
float *local = create1DArray(chunk);
float local_Avg=0.0;
float sum_Of_Avg =0.0;
float local_Var = 0.0;
float stdDev=0.0;


for (i = 0; i < chunk; i++)
    local[i] = 1.0;

if (rank == MASTER) {
         array = create1DArray(N); // only MASTER process stores array data
         for (i = 0; i < N; i++)
              array[i] = 1.0 *i ;

         //printArray(array, N);
    }
MPI_Barrier(MPI_COMM_WORLD);
//printf("rank %d printing bfore scatter...\n", rank);
//printArray(local, chunk);
MPI_Barrier(MPI_COMM_WORLD);
MPI_Scatter(array, chunk, MPI_INT, local, chunk, MPI_INT, MASTER, MPI_COMM_WORLD);
//printf("rank %d printing...\n", rank);
//printArray(local, chunk);

for (i = 0; i < chunk; i++) {
  local_Avg+= local[i];
}
local_Avg /= (float)chunk; //Every processor calculated their own average.
//printf("Rank = %d\tOrtalama %f\n",rank, local_Avg);
MPI_Barrier(MPI_COMM_WORLD);
MPI_Reduce(&local_Avg, &sum_Of_Avg, 1, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
//With MPI_Reduce all slaves sent their calculated average value to the MASTER
//and this values summed.
if(rank==MASTER){
  sum_Of_Avg/=size;// after i gathered all values from slaves i need to divide by number of
  //processor that i worked
}
//printf("Rank = %d\tMaster sum_Of_Avg before BroadCast = %f\n", rank,sum_Of_Avg);
MPI_Bcast(&sum_Of_Avg, 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
//printf("Rank = %d\tMaster sum_Of_Avg after BroadCast = %f\n", rank,sum_Of_Avg);

for (i = 0; i < chunk; i++) {
  local_Var += (local[i]-sum_Of_Avg)*(local[i]-sum_Of_Avg);
}
MPI_Reduce(&local_Var, &stdDev, 1, MPI_FLOAT, MPI_SUM, MASTER, MPI_COMM_WORLD);
MPI_Barrier(MPI_COMM_WORLD);
if(rank==MASTER){
  stdDev /= (N-1); //I'm dividing here because if i do this section in every processor
  //i will lose 1 divider per processor.
  stdDev /= sqrt(stdDev);
  printf("Calculated Standard Deviation  %f\n", stdDev);
}

MPI_Barrier(MPI_COMM_WORLD);
t2 = MPI_Wtime(); // stop clock

printf("Rank = %d\tElapsed time = %f sec.\n", rank,t2 - t1);
//I put some printf sections in comment line. You can check the values with these lines.
MPI_Finalize();
}

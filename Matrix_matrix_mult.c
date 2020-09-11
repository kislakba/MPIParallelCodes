#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define N 9
#define MASTER 0
float *create1DArray(int n){
  float *T  = (float *)malloc(n * sizeof(float));
  return T;
  }
void fillArray(float *T, int n){
  srand(time(0));
  for (int i = 0; i<n; i++)
    T[i] = (float)rand()/(float)(RAND_MAX);
}
void fillArrayWZ(float *T, int n){
  srand(time(0));
  for (int i = 0; i<n; i++)
    T[i] =0;
}
void printArray(float *T, int n1, int n2 ){
  for(int i = 0; i<n1; i++){
    for (int j = 0; j < n2; j++) {
      printf("%.3f \t", T[(i*n2) + j]);
    }
    printf("\n" );
  }
}
void printArrayInt(int *T, int n1, int n2 ){
  for(int i = 0; i<n1; i++){
    for (int j = 0; j < n2; j++) {
      printf("%d \t", T[(i*n2) + j]);
    }
    printf("\n" );
  }
}
void multMiniMatrixes(float *a_loc, float *b_loc, int n1, int n2,float *local){
  float sum;
  int row = 0;
  for (int i = 0; i < n1 *n2; i++) {
    row = i / n1;
    sum = 0.0;
    #pragma omp parallel for schedule(dynamic) reduction(+:sum)
    for (int j = 0; j < n2; j++) {
      sum += a_loc[(n1*row)+j] * b_loc[(j*n2)+(i%n2)];
    }
    local[i]+=sum;
  }
}
int main(int argc, char *argv[])
{
    int rank, size, provided;
    MPI_Comm comm2D, commCol, commRow;
    double tcomm1, tcomm2, tcal1, tcal2,ttime1,ttime2, tget1;
    float *A, *B, *C;
    float *A_local, *B_local, *C_local;
    float *local, *a_temp_loc, *b_temp_loc;
    int *sendcounts;//For scatterv
    int *displs; //For scatterv
    int nrow, mcol, i, lastrow, p, root;
  	int Iam, id2D, colID,rowID, ndim;
    int sourceRow, destRow, sourceCol, destCol;
  	int coordsCol1D[2],coordsRow1D[2], coords2D[2], dims[2];
  	int belongs[2], periods[2], reorder;
    MPI_Init_thread(NULL,NULL, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;
    ttime1 = MPI_Wtime();//total time
    int sqrtOfSize = sqrt(size); // i have to know bounds, ftr i calculated this
    nrow = sqrtOfSize; mcol = sqrtOfSize; ndim = 2;
  	root = 0; periods[0] = 1; periods[1] = 1; reorder = 1;
    /* create cartesian topology for processes */
  	dims[0] = nrow;		/* number of rows */
  	dims[1] = mcol;		/* number of columns */
  	MPI_Cart_create(MPI_COMM_WORLD, ndim, dims, periods, reorder, &comm2D);
  	MPI_Comm_rank(comm2D, &id2D);
  	MPI_Cart_coords(comm2D, id2D, ndim, coords2D);
    //end of cartesian
    /* Create 1D column subgrids */
    belongs[0] = 1;		/* this dimension belongs to subgrid */
    belongs[1] = 0;
    MPI_Cart_sub(comm2D, belongs, &commCol);
    MPI_Comm_rank(commCol, &colID);
    MPI_Cart_coords(commCol, colID, 1, coordsCol1D);
    MPI_Barrier(MPI_COMM_WORLD);
    /*End of the seperating to subgrids ==> ColumnComm*/
    /* Create 1D Row subgrids */
    belongs[0] = 0;		/* this dimension belongs to subgrid */
    belongs[1] = 1;
    MPI_Cart_sub(comm2D, belongs, &commRow);
    MPI_Comm_rank(commRow, &rowID);
    MPI_Cart_coords(commRow, rowID, 1, coordsRow1D);
    MPI_Barrier(MPI_COMM_WORLD);
    /*End of the seperating to subgrids ==> RowComm*/
    //matrix datatype
    MPI_Datatype matrixType, newMatrixType;
    int blocklength = sqrtOfSize , stride = N, count = sqrtOfSize; //for blocklength im just
    //taking little matrix's x axis's lenght
    MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &matrixType);
    MPI_Type_commit(&matrixType);
    MPI_Type_create_resized(matrixType, 0, 1*sizeof(float), &newMatrixType);
    MPI_Type_commit(&newMatrixType);
    //end of creating datatype
    tget1 = MPI_Wtime();
    A_local=create1DArray(count * count);
    B_local=create1DArray(count * count);
    C_local=create1DArray(count * count);
    a_temp_loc = create1DArray(count* count);
    b_temp_loc = create1DArray(count*count);
    local = create1DArray(count* count);
    fillArrayWZ(local, count*count);
    if(rank == MASTER){
      A = create1DArray(N * N);
      fillArray(A, N * N);
      B = create1DArray(N * N);
      fillArray(B, N * N );
      C = create1DArray(N * N);
      fillArrayWZ(C, N * N );
    }
    tcomm1 = MPI_Wtime(); //comm. time
    sendcounts = malloc(sizeof(int)*size);
    displs = malloc(sizeof(int)*size);
    for (int i = 0; i < sqrtOfSize; i++) {
        for (int j = 0; j < sqrtOfSize; j++) {
          sendcounts[(i*sqrtOfSize)+j] = 1;
          displs[(i*sqrtOfSize)+j] = (blocklength*blocklength*sqrtOfSize*i)+(blocklength*j);//for scatterv.
        };
    }
    MPI_Scatterv(A, sendcounts, displs, newMatrixType, A_local,
       count*count, MPI_FLOAT, MASTER, comm2D); //i scattered A matrixs
    MPI_Scatterv(B, sendcounts, displs, newMatrixType, B_local,
       count*count, MPI_FLOAT, MASTER, comm2D);

    MPI_Barrier(MPI_COMM_WORLD);
    tcal1 = MPI_Wtime();//calculation time
    multMiniMatrixes(A_local, B_local, count, count, local);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < sqrtOfSize-1; i++) {
      MPI_Cart_shift(commRow, 1,1+i ,&sourceRow, &destRow);
      MPI_Cart_shift(commCol, 0,1+i ,&sourceCol, &destCol);
      MPI_Sendrecv(A_local,count*count,MPI_FLOAT,destRow,destRow,a_temp_loc
                  ,count*count,MPI_FLOAT,sourceRow,coordsRow1D[0],commRow, &status);
      MPI_Sendrecv(B_local,count*count,MPI_FLOAT,destCol,destCol,b_temp_loc
                  ,count*count,MPI_FLOAT,sourceCol,coordsCol1D[0],commCol, &status);
      multMiniMatrixes(a_temp_loc, b_temp_loc, count, count,local);
      }
    MPI_Barrier(MPI_COMM_WORLD);
    tcal2 = MPI_Wtime();//calculation time
    MPI_Gatherv(local, count*count,MPI_FLOAT ,C,
       sendcounts,displs,newMatrixType , MASTER, comm2D);
    tcomm2 = MPI_Wtime(); //comm. time
    MPI_Barrier(MPI_COMM_WORLD);
    ttime2 = MPI_Wtime();//total time
    if(rank== 0 ){
      printf("total time \t comm time \t calculation time \n");
      printf("%f \t %f \t %f \n",(ttime2-ttime1), (tcomm2-tcomm1)-(tcal2-tcal1),(tcal2-tcal1));
    }
    MPI_Finalize();
    return 0;
}

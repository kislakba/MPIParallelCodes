# Some operational codes with MPI

My codes about MPI library. I am taking Parallel Programming course and this codes generally my home works.


# Files in this project

### Standard Deviation 
Array [Standard Deviation ](https://github.com/kislakba/MPIParallelCodes/blob/master/StdDevParl.c) calculator with multiple processor.

### Matrix Vector Multiplication (Row Based) 
[Matrix Vector Multiplication ](https://github.com/kislakba/MPIParallelCodes/blob/master/MatrixVectorMultRowBased.c). Code is dividing the main matrix to the rows and sharing with other processors. 

### Matrix Vector Multiplication (Column Based) 
[Matrix Vector Multiplication ](https://github.com/kislakba/MPIParallelCodes/blob/master/MatrixVectorMultColumnBased.c). Code is dividing the main matrix to the columns and sharing with other processors. 

### Matrix Matrix Multiplication (MPI- Open-MPI)
[Matrix Matrix Multiplication](https://github.com/kislakba/MPIParallelCodes/blob/master/Matrix_matrix_mult.c). This project split the matrix into chunks of cores. To MPI you have to declare perfect square numbers for cores. It's very inefficent way for this but I made it for learn all basics of MPI and Open MPI.    

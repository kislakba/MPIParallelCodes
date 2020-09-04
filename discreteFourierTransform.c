#include <stdio.h>
#include <complex.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.141592654
double complex *create1DComplexArray(int n) {
     double complex *T = (double complex *)malloc(n * sizeof(double complex));
     return T;
}
double *create1DDoubleArray(int n) {
     double *T = (double *)malloc(n * sizeof(double ));
     return T;
}
void printComplexArray(double complex *T, int n) {
     int i;
     for (i = 0; i < n; i++)
          printf("%d. iteration of DFT = %.3f %+.3fi\n", i + 1, creal(T[i]), cimag(T[i]));

     puts("");
}
void printDoubleArray(double *T, int n) {
     int i;
     for (i = 0; i < n; i++)
          printf(" %.3f\t", T[i]);

     puts("");
}
double r2()
{
    return (double)rand() / (double)RAND_MAX ;
}
void fillArray(double *T, int n) {
     int i;
     for (i = 0; i < n; i++)
          T[i]=r2();

     }
double complex *dft(double *points, int size){

  double complex *result =create1DComplexArray(size);
  double complex temp;
  int n,k;
  double arg;
  double cosarg, sinarg;
  for (k = 0; k < size; k++) {
    result[k] = 0   + 0*I;
    arg = -2.0 * PI * k / (double)size;
    for (n = 0; n < size; n++) {
      cosarg = points[n]*cos(n * arg);
      sinarg = points[n]*sin(n * arg);
      temp = cosarg + sinarg * I;
      result[k] += temp;
    }
  }
  return result;
}
int main(void) {
  double *points = create1DDoubleArray(20);
  fillArray(points, 20);
  printDoubleArray(points, 20);
  double complex *res = dft(points, 20);
  printComplexArray(res, 20);

  return 0;
}

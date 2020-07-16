#include <cublas_v2.h>
#include <curand.h>

#ifdef FLOAT_AS_DOUBLE

typedef double Float;
#define cublasGeam cublasDgeam
#define cublasGemm cublasDgemm
#define curandGenerateUniform curandGenerateUniformDouble

#else

typedef float Float;
#define cublasGeam cublasSgeam
#define cublasGemm cublasSgemm

#endif // FLOAT_AS_DOUBLE

void ata(double *A, double *C, int lda, int ldc, int XA, int XC, int YA, int YC, int depth);

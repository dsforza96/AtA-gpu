#include <cublas_v2.h>

#ifdef FLOAT_AS_DOUBLE
typedef double Float;
#define cublasGeam cublasDgeam
#define cublasGemm cublasDgemm
#define CUTOFF 256

#else
typedef float Float;
#define cublasGeam cublasSgeam
#define cublasGemm cublasSgemm
#define CUTOFF 1536

#endif  // FLOAT_AS_DOUBLE

void ata(Float *A, Float *C, int lda, int ldc, int XA, int XC, int YA, int YC, int depth);

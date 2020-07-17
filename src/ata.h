#include <cublas_v2.h>
#include <curand.h>

#ifdef FLOAT_AS_DOUBLE
typedef double Float;
#define cublasGeam cublasDgeam
#define cublasGemm cublasDgemm
#define curandGenerateUniform curandGenerateUniformDouble
#define CUTOFF 256

#else
typedef float Float;
#define cublasGeam cublasSgeam
#define cublasGemm cublasSgemm
#define CUTOFF 2048

#endif // FLOAT_AS_DOUBLE

void ata(Float *A, Float *C, int lda, int ldc, int XA, int XC, int YA, int YC, int depth);

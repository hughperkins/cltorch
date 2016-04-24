#ifndef THCL_BLAS_INC
#define THCL_BLAS_INC

#include "THClGeneral.h"

//class THClTensor;
struct THClTensor;
class CLWrapper;

typedef struct THClBlasState {
//  cublasHandle_t* handles;
//  cublasHandle_t* current_handle;
//  int n_devices;
} THClBlasState;

/* Level 1 */
THCL_API void THClBlas_swap(THClState *state, long n, float *x, long incx, float *y, long incy);
THCL_API void THClBlas_scal(THClState *state, long n, float a, float *x, long incx);
THCL_API void THClBlas_copy(THClState *state, long n, float *x, long incx, float *y, long incy);
THCL_API void THClBlas_axpy(THClState *state, long n, float a, float *x, long incx, float *y, long incy);
THCL_API float THClBlas_dot(THClState *state, long n, 
    CLWrapper *xwrapper, long xoffset, long incx, 
    CLWrapper *ywrapper, long yoffset, long incy);

/* Level 2 */
THCL_API void THClBlas_gemv(THClState *state, char trans, long m, long n, float alpha,
  THClTensor *a, long lda, 
  THClTensor *x, long incx, 
  float beta,
  THClTensor *y, long incy);

THCL_API void THClBlas_ger(THClState *state, long m, long n, float alpha, 
    THClTensor *x, long incx,
    THClTensor *y, long incy,
    THClTensor *a, long lda);

/* Level 3 */
THCL_API void THClBlas_gemm(THClState *state, char transa, char transb, 
  long m, long n, long k, float alpha,
  THClTensor *a, long lda, THClTensor *b, long ldb, float beta, THClTensor *c, long ldc);

THCL_API void THClBlas_gemm2(THClState *state, char orderchar, char transa, char transb, 
  long m, long n, long k, float alpha,
  THClTensor *a, long lda, THClTensor *b, long ldb, float beta, THClTensor *c, long ldc);

THCL_API void THClBlas_gemmBatched(THClState *state, char transa, char transb, long m, long n, long k,
                                    float alpha, CLWrapper *aWrapper, long lda, CLWrapper *bWrapper, long ldb,
                                    float beta, CLWrapper *cWrapper, long ldc, long batchCount);

#endif


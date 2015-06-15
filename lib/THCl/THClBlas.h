#ifndef THCL_BLAS_INC
#define THCL_BLAS_INC

#include "THClGeneral.h"

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
THCL_API void THClBlas_gemv(THClState *state, char trans, long m, long n, float alpha, CLWrapper *awrapper, long aoffset, long lda, 
     CLWrapper *xwrapper, long xoffset, long incx, 
      float beta, CLWrapper *ywrapper, long yoffset, long incy);
//THCL_API void THClBlas_ger(THClState *state, long m, long n, float alpha, float *x, long incx, float *y, long incy, float *a, long lda);
THCL_API void THClBlas_ger(THClState *state, long m, long n, float alpha, 
    CLWrapper *xwrap, long x_offset, long incx,
    CLWrapper *ywrap, long y_offset, long incy,
    CLWrapper *awrap, long a_offset, long lda);

/* Level 3 */
THCL_API void THClBlas_gemm(THClState *state, char transa, char transb, long m, long n, long k, float alpha, CLWrapper *aWrap, long offseta, long lda, CLWrapper *bWrap, long offsetb, long ldb, float beta, CLWrapper *cWrap, long offsetc, long ldc);
THCL_API void THClBlas_gemmBatched(THClState *state, char transa, char transb, long m, long n, long k,
                                    float alpha, CLWrapper *aWrapper, long lda, CLWrapper *bWrapper, long ldb,
                                    float beta, CLWrapper *cWrapper, long ldc, long batchCount);

#endif


#ifndef THCL_BLAS_INC
#define THCL_BLAS_INC

#include "THClGeneral.h"

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
THCL_API float THClBlas_dot(THClState *state, long n, float *x, long incx, float *y, long incy);

/* Level 2 */
THCL_API void THClBlas_gemv(THClState *state, char trans, long m, long n, float alpha, float *a, long lda, float *x, long incx, float beta, float *y, long incy);
THCL_API void THClBlas_ger(THClState *state, long m, long n, float alpha, float *x, long incx, float *y, long incy, float *a, long lda);

/* Level 3 */
THCL_API void THClBlas_gemm(THClState *state, char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc);
THCL_API void THClBlas_gemmBatched(THClState *state, char transa, char transb, long m, long n, long k,
                                    float alpha, const float *a[], long lda, const float *b[], long ldb,
                                    float beta, float *c[], long ldc, long batchCount);

#endif

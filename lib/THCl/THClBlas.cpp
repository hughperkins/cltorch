#include <iostream>

#include "THClBlas.h"
#include "THClGeneral.h"
#include "THClTensor.h"

#include "util/easycl_stringhelper.h"
#include "EasyCL.h"
#include <clBLAS.h>

using namespace std;

clblasTranspose convertTransToClblasOperation(char trans) {
  if (trans == 't') return clblasTrans;
  else if (trans == 'n') return clblasNoTrans ;
  else if (trans == 'c') return clblasConjTrans;
  else {
    THError("trans must be one of: t, n, c");
    return clblasTrans;
  }
  THError("Not implemented");
}

void THClBlas_init(THClState *state, int devices, int device)
{
//  THClBlasState *blas_state = state->blasState;
//  blas_state->handles = (cublasHandle_t *)malloc(devices * sizeof(cublasHandle_t));
//  for (int i = 0; i < devices; i++) {
//    // Create handle on each device:
//    ClSetDevice(i);
//    cublasCreate(&blas_state->handles[i]);
//  }

//  // Set current handle:
//  blas_state->current_handle = &blas_state->handles[device];
//  blas_state->n_devices = devices;

//  // Restore device:
//  ClSetDevice(device);
    THError("Not implemented");
}

void THClBlas_shutdown(THClState *state)
{
//  THClBlasState *blas_state = state->blasState;
//  for (int i = 0; i < blas_state->n_devices; i++) {
//    cublasDestroy(blas_state->handles[i]);
//  }
//  free(blas_state->handles);
    THError("Not implemented");
}

void THClBlas_setHandle(THClState *state, int device)
{
//  THClBlasState *blas_state = state->blasState;
//  blas_state->current_handle = &blas_state->handles[device];
    THError("Not implemented");
}

//void THClBlas_setStream(THClState *state, int device, ClStream_t stream)
//{
////  THCublasCheck(cublasSetStream(state->blasState->handles[device], stream));
//    THError("Not implemented");
//}

void THClBlas_swap(THClState *state, long n, float *x, long incx, float *y, long incy)
{
//  if(n == 1)
//  {
//    incx = 1;
//    incy = 1;
//  }

//  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
//  {
//    int i_n = (int)n;
//    int i_incx = (int)incx;
//    int i_incy = (int)incy;
//    THCublasCheck(cublasSswap(*state->blasState->current_handle, i_n, x, i_incx, y, i_incy));
//    return;
//  }
//  THError("Cublas_swap only supports n, incx and"
//          " incy upto signed integer limits: %d", INT_MAX);
    THError("Not implemented");
}

void THClBlas_scal(THClState *state, long n, float a, float *x, long incx)
{
 // if(n == 1)
//    incx = 1;

//  if( (n <= INT_MAX) && (incx <= INT_MAX) )
//  {
//    int i_n = (int)n;
//    int i_incx = (int)incx;
//    THCublasCheck(cublasSscal(*state->blasState->current_handle, i_n, &a, x, i_incx));
//    return;
//  }
//  THError("Cublas_scal only supports n and incx "
//          "upto signed integer limits: %d", INT_MAX);
    THError("Not implemented");
}

void THClBlas_copy(THClState *state, long n, float *x, long incx, float *y, long incy)
{
//  if(n == 1)
//  {
//    incx = 1;
//    incy = 1;
//  }

//  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
//  {
//    int i_n = (int)n;
//    int i_incx = (int)incx;
//    int i_incy = (int)incy;
//    THCublasCheck(cublasScopy(*state->blasState->current_handle, i_n, x, i_incx, y, i_incy));
//    return;
//  }

//  THError("Cublas_copy only supports n, incx and incy "
//          "upto signed integer limits: %d", INT_MAX);
    THError("Not implemented");
}

void THClBlas_axpy(THClState *state, long n, float a, float *x, long incx, float *y, long incy)
{
 //   if(n == 1)
//  {
//    incx = 1;
//    incy = 1;
//  }

//  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
//  {
//    int i_n = (int)n;
//    int i_incx = (int)incx;
//    int i_incy = (int)incy;
//    THCublasCheck(cublasSaxpy(*state->blasState->current_handle, i_n, &a, x, i_incx, y, i_incy));
//    return;
//  }

//  THError("Cublas_axpy only supports n, incx and incy "
//          "upto signed integer limits: %d", INT_MAX);
    THError("Not implemented");
}

float THClBlas_dot(THClState *state, long n, 
    CLWrapper *xwrapper, long xoffset, long incx, 
    CLWrapper *ywrapper, long yoffset, long incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    float result;

    cl_int err;

//    err = clblasSetup();
//    if (err != CL_SUCCESS) {
//        THError("clblasSetup() failed with %d", err);
//    }
    
    CLWrapper *resultWrapper = THClState_getCl(state)->wrap( 1, &result );
    float *scratch = new float[i_n];
    CLWrapper *scratchWrapper = THClState_getCl(state)->wrap(i_n, scratch);
    scratchWrapper->createOnDevice();
    resultWrapper->createOnDevice();

    cl_event event = NULL;
    err = clblasSdot( i_n, resultWrapper->getBuffer(), 0, 
          xwrapper->getBuffer(), xoffset, i_incx, 
          ywrapper->getBuffer(), yoffset, i_incy, 
          scratchWrapper->getBuffer(),
          1, THClState_getCl(state)->queue, 0, NULL, &event);
//    THCublasCheck(cublasSdot(*state->blasState->current_handle, i_n, x, i_incx, y, i_incy, &result));
//    ClDeviceSynchronize();
    if (err != CL_SUCCESS) {
        THError("clblasSdot() failed with %d", err);
    }
    else {
        /* Wait for calculations to be finished. */
//        err = clWaitForEvents(1, &event);
    }
    resultWrapper->copyToHost();
//    resultWrapper->markDeviceDirty();

    /* Finalize work with clblas. */
//    clblasTeardown();


    delete resultWrapper;
    delete scratchWrapper;
    delete[] scratch;
    return result;
  }
  THError("Cublas_dot only supports n, incx and incy "
          "upto signed integer limits: %d", INT_MAX);
    THError("Not implemented");
  return -1;
}

/* Level 2 */
void THClBlas_gemv(THClState *state, char trans, long m, long n, float alpha, 
    THClTensor *a, long lda, 
    THClTensor *x, long incx, 
    float beta,
     THClTensor *y, long incy)
{
  CLWrapper *awrapper = THClTensor_wrapper(state, a);
  CLWrapper *xwrapper = THClTensor_wrapper(state, x);
  CLWrapper *ywrapper = THClTensor_wrapper(state, y);
  long aoffset = THClTensor_storageOffset(state, a);
  long xoffset = THClTensor_storageOffset(state, x);
  long yoffset = THClTensor_storageOffset(state, y);


  if(n == 1)
    lda = m;

  clblasTranspose op = convertTransToClblasOperation(trans);

  if( (m <= INT_MAX) && (n <= INT_MAX) &&
      (lda > 0) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    cl_int err;

//    err = clblasSetup();
//    if (err != CL_SUCCESS) {
//        THError("clblasSetup() failed with %d", err);
//    }

    cl_event event = NULL;
    err = clblasSgemv(clblasColumnMajor, op, i_m, i_n, alpha,
          awrapper->getBuffer(), aoffset, i_lda, 
          xwrapper->getBuffer(), xoffset, i_incx, 
          beta,
          ywrapper->getBuffer(), yoffset, i_incy, 
          1, THClState_getCl(state)->queue, 0, NULL, &event);
//    THCublasCheck(cublasSgemv(*state->blasState->current_handle, op, i_m, i_n, &alpha, a, i_lda, x, i_incx, &beta, y, i_incy));
    if (err != CL_SUCCESS) {
        THError("clblasSdot() failed with %d", err);
    }
    else {
        /* Wait for calculations to be finished. */
//        err = clWaitForEvents(1, &event);
    }

    /* Finalize work with clblas. */
//    clblasTeardown();

    ywrapper->markDeviceDirty();
    return;
  }
  THError("Cublas_gemv only supports m, n, lda, incx, incy"
          "in the range 0 < [val] <= %d", INT_MAX);
//    THError("Not implemented");
}

void THClBlas_ger(THClState *state, long m, long n, float alpha, 
    THClTensor *x, long incx,
    THClTensor *y, long incy,
    THClTensor *a, long lda)
{
  CLWrapper *xwrap = THClTensor_wrapper(state, x);
  CLWrapper *ywrap = THClTensor_wrapper(state, y);
  CLWrapper *awrap = THClTensor_wrapper(state, a);
  long x_offset = THClTensor_storageOffset(state, x);
  long y_offset = THClTensor_storageOffset(state, y);
  long a_offset = THClTensor_storageOffset(state, a);

  if(n == 1)
    lda = m;

  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
    {
      int i_m = (int)m;
      int i_n = (int)n;
      int i_lda = (int)lda;
      int i_incx = (int)incx;
      int i_incy = (int)incy;

      cl_int err;

//    THClState_getCl(state)->finish();
//      err = clblasSetup();
//      if (err != CL_SUCCESS) {
//          THError("clblasSetup() failed with %d", err);
//      }
//    THClState_getCl(state)->finish();

      if(!awrap->isOnDevice()) {
        awrap->createOnDevice();
      }

      cl_event event = NULL;
      err = clblasSger(clblasColumnMajor, i_m, i_n, alpha,
                       xwrap->getBuffer(), x_offset, i_incx,
                       ywrap->getBuffer(), y_offset, i_incy,
                       awrap->getBuffer(), a_offset, i_lda,
                       1, (THClState_getCl(state)->queue), 0, NULL, &event);
      if (err != CL_SUCCESS) {
        throw runtime_error("clblasSger() failed with " + easycl::toString(err));
        THError("clblasSger() failed with %d", err);
      }
      else {
          /* Wait for calculations to be finished. */
//          err = clWaitForEvents(1, &event);
      }
//    THClState_getCl(state)->finish();
      awrap->markDeviceDirty();

//      clblasTeardown();
//    THClState_getCl(state)->finish();

//      THCublasCheck(cublasSger(*state->blasState->current_handle, i_m, i_n, &alpha, x, i_incx, y, i_incy, a, i_lda));
      return;
    }
  THError("Cublas_ger only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}

void adjustLd(char transa, char transb, long m, long n, long k, long *lda, long *ldb, long *ldc)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    *ldc = m;

  if(transa_)
  {
    if(m == 1)
      *lda = k;
  }
  else
  {
    if(k == 1)
      *lda = m;
  }

  if(transb_)
  {
    if(k == 1)
      *ldb = n;
  }
  else
  {
    if(n == 1)
      *ldb = k;
  }
}

/* Level 3 */
void THClBlas_gemm(THClState *state, char transa, char transb, long m, long n, long k, float alpha, 
  THClTensor *a, long lda, THClTensor *b, long ldb, float beta, 
  THClTensor *c, long ldc)
{
  CLWrapper *aWrapper = THClTensor_wrapper(state, a);
  CLWrapper *bWrapper = THClTensor_wrapper(state, b);
  CLWrapper *cWrapper = THClTensor_wrapper(state, c);
  long offseta = THClTensor_storageOffset(state, a);
  long offsetb = THClTensor_storageOffset(state, b);
  long offsetc = THClTensor_storageOffset(state, c);

  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  clblasTranspose opa = convertTransToClblasOperation(transa);
  clblasTranspose opb = convertTransToClblasOperation(transb);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    cl_int err;

//    err = clblasSetup();
//    if (err != CL_SUCCESS) {
//        THError("clblasSetup() failed with %d", err);
//    }

    if( !aWrapper->isOnDevice() ) {
      aWrapper->createOnDevice();
    }
    if( !bWrapper->isOnDevice() ) {
      bWrapper->createOnDevice();
    }
    if( !cWrapper->isOnDevice() ) {
      cWrapper->createOnDevice();
    }

    cl_event event = NULL;
    err = clblasSgemm(clblasColumnMajor, opa, opb, i_m, i_n, i_k,
                         alpha, aWrapper->getBuffer(), offseta, i_lda,
                         bWrapper->getBuffer(), offsetb, i_ldb, beta,
                         cWrapper->getBuffer(), offsetc, i_ldc,
                         1, THClState_getCl(state)->queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        THError("clblasSgemm() failed with %d", err);
    }
    else {
//        err = clWaitForEvents(1, &event);
    }

//    clblasTeardown();

//    THCublasCheck(cublasSgemm(*state->blasState->current_handle, opa, opb, i_m, i_n, i_k, &alpha, a, i_lda, b, i_ldb, &beta, c, i_ldc));
    return;
  }
  THError("Clblas_gemm only supports m, n, k, lda, ldb, ldc"
          "with the bound [val] <= %d", INT_MAX);
}

void THClBlas_gemmBatched(THClState *state, char transa, char transb, long m, long n, long k,
                            float alpha, const float *a[], long lda, const float *b[], long ldb,
                            float beta, float *c[], long ldc, long batchCount)
{
//  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )
//  {
//    THError("Cublas_gemm only supports m, n, k, lda, ldb, ldc, batchCount"
//            "with the bound [val] <= %d", INT_MAX);
//  }

//  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
//  cublasOperation_t opa = convertTransToCublasOperation(transa);
//  cublasOperation_t opb = convertTransToCublasOperation(transb);

//  THCublasCheck(cublasSgemmBatched(*state->blasState->current_handle,
//                                   opa, opb, (int)m, (int)n, (int)k,
//                                   &alpha, a, (int)lda, b, (int)ldb, &beta, c, (int)ldc,
//                                   (int)batchCount));
    THError("Not implemented");
}


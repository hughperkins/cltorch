#include <iostream>

#include "THClTensorMath.h"
#include "THClGeneral.h"
#include "THClBlas.h"
#include "THClTensorCopy.h"
//#include "THClTensorRandom.h"
#include "THClApply.h"
//#include "THClReduce.cuh"

using namespace std;

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

float THClTensor_dot(THClState *state, THClTensor *self, THClTensor *src)
{
  THAssert(THClTensor_checkGPU(state, 2, self, src));
  THArgCheck(THClTensor_nElement(state, self) == THClTensor_nElement(state, src), 2, "sizes do not match");

  {
    self = THClTensor_newContiguous(state, self);
    src = THClTensor_newContiguous(state, src);

    float result = THClBlas_dot(state,
                                  THClTensor_nElement(state, self),
                                  THClTensor_wrapper(state, self), 
                                  THClTensor_storageOffset(state, self), 
                                  1,
                                  THClTensor_wrapper(state, src), 
                                  THClTensor_storageOffset(state, src), 
                                  1);
    THClTensor_free(state, src);
    THClTensor_free(state, self);

    return result;
  }
//  THError("Not implemented");
//  return 0;
}

void THClTensor_addmv(THClState *state, THClTensor *r_, float beta, THClTensor *t, float alpha, THClTensor *mat, THClTensor *vec)
{
  THAssert(THClTensor_checkGPU(state, 4, r_, t, mat, vec));
  if( (mat->nDimension != 2) || (vec->nDimension != 1) )
    THError("matrix and vector expected");

  if( mat->size[1] != vec->size[0] )
    THError("size mismatch");

  if(t->nDimension != 1)
    THError("size mismatch");

  if(t->size[0] != mat->size[0])
    THError("size mismatch");

  if(r_ != t)
  {
    THClTensor_resizeAs(state, r_, t);
    THClTensor_copy(state, r_, t);
  }

  if(mat->stride[0] == 1)
  {
    THClBlas_gemv(state, 'n', mat->size[0], mat->size[1],
                    alpha, 
                    mat, 
                    mat->stride[1],
                    vec, 
                    vec->stride[0],
                    beta, 
                    r_,
                    r_->stride[0]);
  }
  else if(mat->stride[1] == 1)
  {
    THClBlas_gemv(state, 't',  mat->size[1], mat->size[0],
                    alpha, 
                    mat,
                    mat->stride[0],
                    vec,
                    vec->stride[0],
                    beta,
                    r_,
                    r_->stride[0]);
  }
  else
  {
    THClTensor *cmat = THClTensor_newContiguous(state, mat);

    THClBlas_gemv(state, 't',  mat->size[1], mat->size[0],
                    alpha, 
                    cmat,
                    cmat->stride[0],
                    vec,
                    vec->stride[0],
                    beta,
                    r_,
                    r_->stride[0]);

    THClTensor_free(state, cmat);
  }
//  THError("Not implemented");
}

void THClTensor_addmm(THClState *state, THClTensor *r_, float beta, THClTensor *t, float alpha, THClTensor *m1, THClTensor *m2)
{
//  throw runtime_error("foo");
  THAssert(THClTensor_checkGPU(state, 4, r_, t, m1, m2));
  char transpose_r, transpose_m1, transpose_m2;
  THClTensor *r__, *m1_, *m2_;

  if( (m1->nDimension != 2) || (m2->nDimension != 2) )
    THError("matrix and matrix expected");

  if(t->nDimension != 2)
    THError("size mismatch");

  if( (t->size[0] != m1->size[0]) || (t->size[1] != m2->size[1]) || (m1->size[1] != m2->size[0]) )
    THError("size mismatch");

  if(t != r_)
  {
    THClTensor_resizeAs(state, r_, t);
    THClTensor_copy(state, r_, t);
  }

  /* r_ */
  if(r_->stride[0] == 1)
  {
    transpose_r = 'n';
    r__ = r_;
  }
  else if(r_->stride[1] == 1)
  {
    THClTensor *swap = m2;
    m2 = m1;
    m1 = swap;
    transpose_r = 't';
    r__ = r_;
  }
  else
  {
    transpose_r = 'n';

    r__ = THClTensor_newWithSize2d(state, t->storage->device, r_->size[1], r_->size[0]);
    THClTensor_copy(state, r__, r_);
    THClTensor_transpose(state, r__, NULL, 0, 1);
  }

  /* m1 */
  if(m1->stride[(transpose_r == 'n' ? 0 : 1)] == 1)
  {
    transpose_m1 = 'n';
    m1_ = m1;
  }
  else if(m1->stride[(transpose_r == 'n' ? 1 : 0)] == 1)
  {
    transpose_m1 = 't';
    m1_ = m1;
  }
  else
  {
    transpose_m1 = (transpose_r == 'n' ? 't' : 'n');
    m1_ = THClTensor_newContiguous(state, m1);
  }

  /* m2 */
  if(m2->stride[(transpose_r == 'n' ? 0 : 1)] == 1)
  {
    transpose_m2 = 'n';
    m2_ = m2;
  }
  else if(m2->stride[(transpose_r == 'n' ? 1 : 0)] == 1)
  {
    transpose_m2 = 't';
    m2_ = m2;
  }
  else
  {
    transpose_m2 = (transpose_r == 'n' ? 't' : 'n');
    m2_ = THClTensor_newContiguous(state, m2);
  }

  /* do the operation */
  THClBlas_gemm2(state,
                'c',
                  transpose_m1,
                  transpose_m2,
                  r__->size[(transpose_r == 'n' ? 0 : 1)],
                  r__->size[(transpose_r == 'n' ? 1 : 0)],
                  m1_->size[(transpose_r == 'n' ? 1 : 0)],
                  alpha,
                  m1_,
                  (transpose_m1 == 'n' ? m1_->stride[(transpose_r == 'n' ? 1 : 0)] : m1_->stride[(transpose_r == 'n' ? 0 : 1)]),
                  m2_,
                  (transpose_m2 == 'n' ? m2_->stride[(transpose_r == 'n' ? 1 : 0)] : m2_->stride[(transpose_r == 'n' ? 0 : 1)]),
                  beta,
                  r__,
                  r__->stride[(transpose_r == 'n' ? 1 : 0)]);

//  r__->storage->wrapper->markDeviceDirty();

  /* free intermediate variables */
  if(m1_ != m1)
    THClTensor_free(state, m1_);

  if(m2_ != m2)
    THClTensor_free(state, m2_);

  if(r__ != r_) {
    THClTensor_freeCopyTo(state, r__, r_);
  }
}

void THClTensor_addr(THClState *state, THClTensor *r_, float beta, THClTensor *t, float alpha, THClTensor *vec1, THClTensor *vec2)
{
  THAssert(THClTensor_checkGPU(state, 4, r_, t, vec1, vec2));
  if( (vec1->nDimension != 1) || (vec2->nDimension != 1) )
    THError("vector and vector expected");

  if(t->nDimension != 2)
    THError("size mismatch");

  if( (t->size[0] != vec1->size[0]) || (t->size[1] != vec2->size[0]) )
    THError("size mismatch");

  if(r_ != t)
  {
    THClTensor_resizeAs(state, r_, t);
    THClTensor_copy(state, r_, t);
  }

  if(beta != 1)
  {
    THClTensor_mul(state, r_, r_, beta);
  }

  if(r_->stride[0] == 1)
  {
    THClBlas_ger(state, vec1->size[0], vec2->size[0],
                   alpha, 
                  vec1, 
                  vec1->stride[0],
                   vec2,
                   vec2->stride[0],
                   r_, 
                    r_->stride[1]);
  }
  else if(r_->stride[1] == 1)
  {
    THClBlas_ger(state, vec2->size[0], vec1->size[0],
                   alpha, 
                  vec2,
                   vec2->stride[0],
                   vec1,
                    vec1->stride[0],
                   r_, 
                    r_->stride[0]);
  }
  else
  {
    THClTensor *cr = THClTensor_newClone(state, r_);

    THClBlas_ger(state, vec2->size[0], vec1->size[0],
                   alpha, 
                  vec2,
                  vec2->stride[0],
                   vec1,
                    vec1->stride[0],
                   cr,
                    cr->stride[0]);

    THClTensor_freeCopyTo(state, cr, r_);
  }
//    THError("Not implemented");
}

void THClTensor_baddbmm(THClState *state, THClTensor *result, float beta, THClTensor *t,
                          float alpha, THClTensor *batch1, THClTensor *batch2) {
//  THAssert(THClTensor_checkGPU(state, 4, result, t, batch1, batch2));
//  THArgCheck(THClTensor_nDimension(state, t) == 3, 4, "expected 3D tensor");
//  THArgCheck(THClTensor_nDimension(state, batch1) == 3, 6, "expected 3D tensor");
//  THArgCheck(THClTensor_nDimension(state, batch2) == 3, 7, "expected 3D tensor");
//  THArgCheck(THClTensor_size(state, t, 0) == THClTensor_size(state, batch1, 0), 6,
//             "equal number of batches expected");
//  THArgCheck(THClTensor_size(state, t, 0) == THClTensor_size(state, batch2, 0), 7,
//             "equal number of batches expected");
//  THArgCheck(THClTensor_size(state, t, 1) == THClTensor_size(state, batch1, 1), 6,
//             "wrong matrix size");
//  THArgCheck(THClTensor_size(state, t, 2) == THClTensor_size(state, batch2, 2), 7,
//             "wrong matrix size");
//  THArgCheck(THClTensor_size(state, batch1, 2) == THClTensor_size(state, batch2, 1), 6,
//             "wrong matrix size");

//  if (t != result) {
//    THClTensor_resizeAs(state, result, t);
//    THClTensor_copy(state, result, t);
//  }

//  bool transpose_result;
//  char transpose_batch1, transpose_batch2;
//  long lda, ldb, ldc;
//  THClTensor *result_, *batch1_, *batch2_;
//  if (result->stride[1] == 1)
//  {
//    transpose_result = false;
//    result_ = result;
//    ldc = result_->stride[2];
//  }
//  else if (result->stride[2] == 1)
//  {
//    transpose_result = true;

//    THClTensor *swap = batch2;
//    batch2 = batch1;
//    batch1 = swap;

//    result_ = result;
//    ldc = result_->stride[1];
//  }
//  else
//  {
//    transpose_result = false;

//    result_ = THClTensor_newWithSize3d(state, result->size[0], result->size[2], result->size[1]);
//    THClTensor_copy(state, result_, result);
//    THClTensor_transpose(state, result_, NULL, 1, 2);

//    ldc = result_->stride[2];
//  }

//  if (batch1->stride[transpose_result ? 2 : 1] == 1)
//  {
//    transpose_batch1 = 'n';
//    batch1_ = batch1;
//    lda = batch1_->stride[transpose_result ? 1 : 2];
//  }
//  else if (batch1->stride[transpose_result ? 1 : 2] == 1)
//  {
//    transpose_batch1 = 't';
//    batch1_ = batch1;
//    lda = batch1_->stride[transpose_result ? 2 : 1];
//  }
//  else
//  {
//    transpose_batch1 = transpose_result ? 'n' : 't';
//    batch1_ = THClTensor_newContiguous(state, batch1);
//    lda = batch1_->stride[1];
//  }

//  if (batch2->stride[transpose_result ? 2 : 1] == 1)
//  {
//    transpose_batch2 = 'n';
//    batch2_ = batch2;
//    ldb = batch2_->stride[transpose_result ? 1 : 2];
//  }
//  else if (batch2->stride[transpose_result ? 1 : 2] == 1)
//  {
//    transpose_batch2 = 't';
//    batch2_ = batch2;
//    ldb = batch2_->stride[transpose_result ? 2 : 1];
//  }
//  else
//  {
//    transpose_batch2 = transpose_result ? 'n' : 't';
//    batch2_ = THClTensor_newContiguous(state, batch2);
//    ldb = batch2_->stride[1];
//  }

//  // Compute pointers to matrices in each batch.
//  long num_batches = result_->size[0];
//  size_t matrices_size = num_batches * sizeof(float*);
//  const float **matrices1 = (const float **)THAlloc(matrices_size);
//  const float **matrices2 = (const float **)THAlloc(matrices_size);
//  float **result_matrices = (float **)THAlloc(matrices_size);
//  for (int i = 0; i < num_batches; ++i)
//  {
//    matrices1[i] = THClTensor_data(state, batch1_) + i * batch1_->stride[0];
//    matrices2[i] = THClTensor_data(state, batch2_) + i * batch2_->stride[0];
//    result_matrices[i] = THClTensor_data(state, result_) + i * result_->stride[0];
//  }

//  // Copy pointers to device.
//  const float **d_matrices1, **d_matrices2;
//  float **d_result_matrices;
//  THClCheck(cudaMalloc(&d_matrices1, matrices_size));
//  THClCheck(cudaMalloc(&d_matrices2, matrices_size));
//  THClCheck(cudaMalloc(&d_result_matrices, matrices_size));

//  THClCheck(cudaMemcpyAsync(d_matrices1, matrices1, matrices_size,
//                              cudaMemcpyHostToDevice, THClState_getCurrentStream(state)));
//  THClCheck(cudaMemcpyAsync(d_matrices2, matrices2, matrices_size,
//                              cudaMemcpyHostToDevice, THClState_getCurrentStream(state)));
//  THClCheck(cudaMemcpyAsync(d_result_matrices, result_matrices, matrices_size,
//                              cudaMemcpyHostToDevice, THClState_getCurrentStream(state)));

//  THClBlas_gemmBatched(
//      state,
//      transpose_batch1,
//      transpose_batch2,
//      result_->size[transpose_result ? 2 : 1],
//      result_->size[transpose_result ? 1 : 2],
//      batch1_->size[transpose_result ? 1 : 2],
//      alpha,
//      d_matrices1, lda,
//      d_matrices2, ldb,
//      beta,
//      d_result_matrices, ldc,
//      num_batches);

//  cudaFree(d_matrices1);
//  cudaFree(d_matrices2);
//  cudaFree(d_result_matrices);
//  THFree(matrices1);
//  THFree(matrices2);
//  THFree(result_matrices);

//  if (batch1_ != batch1)
//    THClTensor_free(state, batch1_);

//  if (batch2_ != batch2)
//    THClTensor_free(state, batch2_);

//  if (result_ != result)
//    THClTensor_freeCopyTo(state, result_, result);
    THError("Not implemented");
}


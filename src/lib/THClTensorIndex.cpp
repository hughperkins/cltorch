#include "THClTensorMath.h"
#include "THClGeneral.h"
//#include "THClBlas.h"
#include "THClTensorCopy.h"
//#include "THClTensorRandom.h"
#include "THClApply.h"
#include "THClReduce.h"
#include "THClKernels.h"

#include <string>
using namespace std;

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

static std::string getKernelTemplate();

void THClTensor_indexCopy_long(THClState *state, THClTensor *res_, int dim, THLongTensor *indices, THClTensor *src)
{
  THAssert(THClTensor_checkGPU(state, 1, res_));

  THClTensor *indices_ = THClTensor_newWithSize1d(state, src->storage->device, indices->size[0]);
  THClTensor_copyLong(state, indices_, indices);

  THClTensor_indexCopy(state, res_, dim, indices_, src);

  THClTensor_free(state, indices_);
}

void THClTensor_indexCopy(THClState *state, THClTensor *res_, int dim, THClTensor *indices, THClTensor *src)
{
  StatefulTimer::timeCheck("THClTensor_indeCopy START");
  THAssert(THClTensor_checkGPU(state, 2, res_, src));
  int *stride_;
  int nIndex = indices->size[0];
  int nRes;

  THArgCheck(indices->nDimension == 1, 3, "expecting vector of indices");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim is out of bounds");
  THArgCheck(src->nDimension > 0, 2, "Source tensor is empty");
  THArgCheck(nIndex == src->size[dim], 4, "length of src.size[dim] is not equal to length of indices");

  src = THClTensor_newContiguous(state, src);
  indices = THClTensor_newContiguous(state, indices);

  nRes = THClTensor_nElement(state, res_);
  dim3 nthreads(16, 16);
  dim3 nblocks(ceil((float)nRes / nIndex / (16*16)));

  stride_ = new int[res_->nDimension];
  EasyCL *cl = src->storage->cl;
  CLWrapper *strideWrapper = cl->wrap(res_->nDimension, stride_);
  for(int i = 0; i < res_->nDimension; i++ ) {
    stride_[i] = res_->stride[i];
  }
  strideWrapper->copyToDevice();

  std::string uniqueName = "THClTensorMathIndex_indexCopy";
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("Apply3 1aa");
  } else {
    TemplatedKernel kernelBuilder(cl);

    kernel = kernelBuilder.buildKernel(uniqueName, "THClTensorIndex.cl",
      getKernelTemplate(), "THClTensor_kernel_indexCopy");
  }

  THClKernels k(state, kernel);
  k.inout(res_);
  k.in(src);
  k.in(strideWrapper);
  k.in(indices);

  k.in((int)(res_->nDimension));
  k.in((int)dim);
  k.in((int)nIndex);

  k.in((int)(THClTensor_nElement(state, src)));
  k.in((int)(res_->size[dim]));

  k.run(nblocks, nthreads);

  delete strideWrapper;
  delete[] stride_;

  THClTensor_free(state, indices);
  THClTensor_free(state, src);
  if(state->addFinish) cl->finish();  
  StatefulTimer::timeCheck("THClTensor_indexCopy END");
}

void THClTensor_indexFill_long(THClState *state, THClTensor *res_, int dim, THLongTensor *indices, float val)
{
  THAssert(THClTensor_checkGPU(state, 1, res_));

  THClTensor *indices_ = THClTensor_newWithSize1d(state, res_->storage->device, indices->size[0]);
  THClTensor_copyLong(state, indices_, indices);

  THClTensor_indexFill(state, res_, dim, indices_, val);

  THClTensor_free(state, indices_);
}

void THClTensor_indexFill(THClState *state, THClTensor *res_, int dim, THClTensor *indices, float val)
{
  StatefulTimer::timeCheck("THClTensor_indexFill START");
  THAssert(THClTensor_checkGPU(state, 1, res_));
  int *stride_;
  long nIndex = indices->size[0];
  long nRes;

  THArgCheck(indices->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < res_->nDimension,4,"Indexing dim is out of bounds");
  THArgCheck(res_->nDimension > 0, 2, "Source tensor is empty");

  nRes = THClTensor_nElement(state, res_) / res_->size[dim] * nIndex;
  indices = THClTensor_newContiguous(state, indices);

  dim3 nthreads(16, 16);
  dim3 nblocks(ceil((float)nRes / nIndex / (16*16)));

  stride_ = new int[res_->nDimension];
  EasyCL *cl = res_->storage->cl;
  CLWrapper *strideWrapper = cl->wrap(res_->nDimension, stride_);
  for(int i = 0; i < res_->nDimension; i++ ) {
    stride_[i] = res_->stride[i];
  }
  strideWrapper->copyToDevice();

  // launch kernel here....
  std::string uniqueName = "THClTensorMathIndex_indexFill";
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("Apply3 1aa");
  } else {
    TemplatedKernel kernelBuilder(cl);

    kernel = kernelBuilder.buildKernel(uniqueName, "THClTensorIndex.cl",
      getKernelTemplate(), "THClTensor_kernel_indexFill");
  }

  THClKernels k(state, kernel);
  k.inout(res_);
  k.in(strideWrapper);
  k.in(indices);
  k.in((int)(res_->nDimension));
  k.in((int)dim);
  k.in((int)nIndex);
  k.in((int)nRes);
  k.in((int)(res_->size[dim]));
  k.in(val);
  k.run(nblocks, nthreads);

  delete strideWrapper;
  delete[] stride_;
  THClTensor_free(state, indices);
  if(state->addFinish) cl->finish();  
  StatefulTimer::timeCheck("THClTensor_indexFill END");
}

void THClTensor_indexSelect_long(THClState *state, THClTensor *res_, THClTensor *src, int dim, THLongTensor *indices)
{
  THAssert(THClTensor_checkGPU(state, 2, res_, src));

  THClTensor *indices_ = THClTensor_newWithSize1d(state, src->storage->device, indices->size[0]);
  THClTensor_copyLong(state, indices_, indices);

  THClTensor_indexSelect(state, res_, src, dim, indices_);

  THClTensor_free(state, indices_);
}

void THClTensor_indexSelect(THClState *state, THClTensor *res_, THClTensor *src, int dim, THClTensor *indices)
{
  StatefulTimer::timeCheck("THClTensor_indexSelect START");
  THAssert(THClTensor_checkGPU(state, 2, res_, src));
  THClTensor *res;
  THLongStorage *newSize;
  int *stride_;
  int nIndex = indices->size[0];
  int nRes;

  THArgCheck(indices->nDimension == 1, 3, "expecting vector of indices");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim is out of bounds");
  THArgCheck(src->nDimension > 0, 2, "Source tensor is empty");

  newSize = THLongStorage_newWithSize(src->nDimension);
  THLongStorage_rawCopy(newSize, src->size);
  newSize->data[dim] = nIndex;
  THClTensor_resize(state, res_, newSize, NULL);
  THLongStorage_free(newSize);

  res = THClTensor_newContiguous(state, res_);
  indices = THClTensor_newContiguous(state, indices);

  EasyCL *cl = src->storage->cl;

  nRes = THClTensor_nElement(state, res);
  dim3 nthreads(16, 16);
  dim3 nblocks(ceil((float)nRes / nIndex / (16*16)));

  stride_ = new int[src->nDimension];
  CLWrapper *strideWrapper = cl->wrap(src->nDimension, stride_);
  for(int i = 0; i < src->nDimension; i++ ) {
    stride_[i] = src->stride[i];
  }
  strideWrapper->copyToDevice();

  // launch kernel here....
  std::string uniqueName = "THClTensorMathIndex_indexSelect";
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
  } else {
    TemplatedKernel kernelBuilder(cl);

    kernel = kernelBuilder.buildKernel(uniqueName, "THClTensorIndex.cl",
      getKernelTemplate(), "THClTensor_kernel_indexSelect");
    StatefulTimer::timeCheck("IndexSelect compiled kernel");
  }

  THClKernels k(state, kernel);
  k.inout(res);
  k.in(src);
  k.in(strideWrapper);
  k.in(indices);

  k.in((int)(src->nDimension));
  k.in((int)dim);
  k.in((int)indices->size[0]);
  k.in((int)nRes);
  k.in((int)src->size[dim]);

  k.run(nblocks, nthreads);

  delete strideWrapper;
  delete[] stride_;

  THClTensor_free(state, indices);
  THClTensor_freeCopyTo(state, res, res_);
  if(state->addFinish) cl->finish();  
  StatefulTimer::timeCheck("THClTensor_indexSelect END");
}

std::string getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClTensorIndex.cl" )
  // ]]]
  // generated using cog, from THClTensorIndex.cl:
  const char * kernelSource =  
  "// from lib/THC/THCTensorIndex.cu:\n" 
  "\n" 
  "kernel void THClTensor_kernel_indexFill(\n" 
  "   global float *tensor_data, int tensor_offset,\n" 
  "  global int* stride,\n" 
  "  global float *index_data, int index_offset,\n" 
  "  int src_nDim,\n" 
  "   int dim, int idx_size, int tensor_size, int size_dim, float val\n" 
  ")\n" 
  "{\n" 
  "  int thread_idx = get_group_id(0) * get_local_size(0) * get_local_size(1) + get_local_id(1) * get_local_size(0) + get_local_id(0);\n" 
  "\n" 
  "  long flat_size = tensor_size / idx_size;\n" 
  "\n" 
  "  if (thread_idx < flat_size)\n" 
  "  {\n" 
  "    long coeff = 0;\n" 
  "    for (int i=0; i<idx_size; i++)\n" 
  "    {\n" 
  "      int leftover = thread_idx;\n" 
  "      int srcIdx = 0;\n" 
  "      for (int d=0; d<src_nDim; d++)\n" 
  "      {\n" 
  "        if (d < dim)\n" 
  "        {\n" 
  "          coeff = leftover / (stride[d] / size_dim);\n" 
  "          leftover -= coeff * (stride[d] / size_dim);\n" 
  "          srcIdx += coeff * stride[d];\n" 
  "        }\n" 
  "        else if (d > dim)\n" 
  "        {\n" 
  "          coeff = leftover / stride[d];\n" 
  "          leftover -= coeff * stride[d];\n" 
  "          srcIdx += coeff * stride[d];\n" 
  "        }\n" 
  "      }\n" 
  "      tensor_data[tensor_offset + srcIdx + (int)((index_data[index_offset + i])-1)*stride[dim] ] = val;\n" 
  "    }\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "kernel void THClTensor_kernel_indexCopy(\n" 
  "   global float *res_data, int res_offset,\n" 
  "   global float *src_data, int src_offset,\n" 
  "   global int* res_stride, global float *index_data, int index_offset,\n" 
  "   int res_nDim, int dim, int idx_size, int src_size, int size_dim\n" 
  ")\n" 
  "{\n" 
  "  int thread_idx = get_group_id(0) * get_local_size(0) * get_local_size(1) + get_local_id(1) * get_local_size(0) + get_local_id(0);\n" 
  "\n" 
  "  long flat_size = src_size / idx_size;\n" 
  "\n" 
  "  if (thread_idx < flat_size)\n" 
  "  {\n" 
  "    long coeff = 0;\n" 
  "    for (int i=0; i<idx_size; i++)\n" 
  "    {\n" 
  "      int leftover = thread_idx;\n" 
  "      int targetIdx = 0;\n" 
  "      int resIdx = 0;\n" 
  "      for (int d=0; d<res_nDim; d++)\n" 
  "      {\n" 
  "        if (d < dim)\n" 
  "        {\n" 
  "          long stride_d = res_stride[d] / size_dim;\n" 
  "          coeff = leftover / stride_d;\n" 
  "          leftover -= coeff * stride_d;\n" 
  "          targetIdx += coeff * stride_d * idx_size;\n" 
  "          resIdx += coeff * res_stride[d];\n" 
  "        }\n" 
  "        else if (d > dim)\n" 
  "        {\n" 
  "          coeff = leftover / res_stride[d];\n" 
  "          leftover -= coeff * res_stride[d];\n" 
  "          targetIdx += coeff * res_stride[d];\n" 
  "          resIdx += coeff * res_stride[d];\n" 
  "        }\n" 
  "      }\n" 
  "      res_data[res_offset + resIdx + ((int)(index_data[index_offset + i])-1)*res_stride[dim] ] = src_data[src_offset + targetIdx + i*res_stride[dim] ];\n" 
  "    }\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "kernel void THClTensor_kernel_indexSelect(\n" 
  "   global float *tensor_data, int tensor_offset, global float *src_data, int src_offset,\n" 
  "  global int* src_stride, global float *index_data, int index_offset,\n" 
  "   int src_nDim, int dim, int idx_size, int tensor_size, int size_dim\n" 
  ")\n" 
  "{\n" 
  "  int thread_idx = get_group_id(0) * get_local_size(0) * get_local_size(1) + get_local_id(1) * get_local_size(0) + get_local_id(0);\n" 
  "\n" 
  "  long flat_size = tensor_size / idx_size;\n" 
  "\n" 
  "  if (thread_idx < flat_size)\n" 
  "  {\n" 
  "    long coeff = 0;\n" 
  "    for (int i=0; i<idx_size; i++)\n" 
  "    {\n" 
  "      int leftover = thread_idx;\n" 
  "      int targetIdx = 0;\n" 
  "      int srcIdx = 0;\n" 
  "      for (int d=0; d<src_nDim; d++)\n" 
  "      {\n" 
  "        if (d < dim)\n" 
  "        {\n" 
  "          long stride_d = src_stride[d] / size_dim;\n" 
  "          coeff = leftover / stride_d;\n" 
  "          leftover -= coeff * stride_d;\n" 
  "          targetIdx += coeff * stride_d * idx_size;\n" 
  "          srcIdx += coeff * src_stride[d];\n" 
  "        }\n" 
  "        else if (d > dim)\n" 
  "        {\n" 
  "          coeff = leftover / src_stride[d];\n" 
  "          leftover -= coeff * src_stride[d];\n" 
  "          targetIdx += coeff * src_stride[d];\n" 
  "          srcIdx += coeff * src_stride[d];\n" 
  "        }\n" 
  "      }\n" 
  "      tensor_data[tensor_offset + targetIdx + i*src_stride[dim] ] = src_data[src_offset + srcIdx + ((int)(index_data[index_offset + i])-1)*src_stride[dim] ];\n" 
  "    }\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}


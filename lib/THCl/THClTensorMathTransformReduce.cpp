// from lib/THC/THCTensorMathTransformReduce.cu:

#include "THClTensorMath.h"
#include "THClGeneral.h"
#include "THClBlas.h"
#include "THClTensorCopy.h"
//#include "THClTensorRandom.h"
#include "THClApply.h"
#include "THClReduce.h"
#include "THClDeviceUtils.h"
#include "THClKernels.h"

#include <string>
using namespace std;

inline long getBlockSize(THClState *state) {
  int blockSize = 1024;
  int maxWorkgroupSize = ((easycl::DeviceInfo *)state->deviceInfoByDevice[state->currentDevice])->maxWorkGroupSize;
  if( blockSize > maxWorkgroupSize ) {
    blockSize = maxWorkgroupSize;
  }
  return blockSize;
}

std::string THClTensorMathTransformReduce_getKernelTemplate();

class HasPairOperator2 {
public:
  virtual std::string pair_operator2() const = 0;
};

class maxvalue_functor : public HasPairOperator2
{
public:
  std::string pair_operator2() const {
    return "if( a.first > b.first ) { return a; } else { return b; }";
  }
};

class minvalue_functor : public HasPairOperator2
{
public:
  std::string pair_operator2() const {
    return "if( a.first < b.first ) { return a; } else { return b; }";
  }
};

void THClTensor_transformReduceOuterDimIndex(THClState *state, THClTensor *tgt1, THClTensor *tgt2,
                                                   THClTensor *src,
                                                    long rdim, float init,
                                                   HasPairOperator2 *binary_op)
{
  unsigned ndim = THClTensor_nDimension(state, src);
  unsigned num_orows = 1;
  for (unsigned dim = 0; dim < rdim; dim++) {
    num_orows *= THClTensor_size(state, src, dim);
  }
  unsigned row_size = THClTensor_size(state, src, rdim);
  unsigned num_irows = 1;
  for (unsigned dim = rdim + 1; dim < ndim; dim++) {
    num_irows *= THClTensor_size(state, src, dim);
  }

  dim3 threads(mymin(getBlockSize(state)/2, num_irows));
  unsigned maxGridDim = getBlockSize(state);
  dim3 grid(mymin(maxGridDim, num_orows), mymin(maxGridDim, THClCeilDiv(num_irows, threads.x())));

  std::string uniqueName = "THClTensorMathTransformReduce_OuterDim_" + binary_op->pair_operator2() + "_" + easycl::toString(init);
  EasyCL *cl = THClState_getCl(state);
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("Apply3 1aa");
  } else {
    TemplatedKernel kernelBuilder(THClState_getCl(state));

    kernelBuilder
      .set("init", init)
      .set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS)
      .set("x_threads", 1)
      .set("y_threads", 1)
      .set("pair_operator2", binary_op->pair_operator2())
    ;

    kernel = kernelBuilder.buildKernel(uniqueName, "THClTensorMathTransformReduce.cl", THClTensorMathTransformReduce_getKernelTemplate(), "THClTensor_kernel_transformReduceOuterDimIndex");
  }

  THClKernels k(state, kernel);
  k.out( tgt1 );
  k.out( tgt2 );
  k.in( src );
  k.in((int)num_orows);
  k.in((int)num_irows);
  k.in((int)row_size);
  k.run(grid, threads);
}

void THClTensor_transformReduceInnermostDimIndex(
  THClState *state, THClTensor *tgt1, THClTensor *tgt2, THClTensor *src,
  float init, HasPairOperator2 *binary_op)
{
  unsigned ndim = THClTensor_nDimension(state, src);
  unsigned num_rows = 1;
  for (unsigned dim = 0; dim < ndim - 1; dim++) {
    num_rows *= THClTensor_size(state, src, dim);
  }
  unsigned row_size = THClTensor_size(state, src, ndim - 1);

  int x_threads = 16;
  int y_threads = 32;
  if( x_threads * y_threads > getBlockSize(state) ) {
    y_threads = getBlockSize(state) / x_threads;
  }
  dim3 threads(x_threads, y_threads);
  dim3 grid(mymin(getBlockSize(state), THClCeilDiv(num_rows, threads.y())));

  std::string uniqueName = "THClTensorMathTransformReduce_InnermostDim_" + binary_op->pair_operator2() + "_" + easycl::toString(init);
  EasyCL *cl = THClState_getCl(state);
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("Apply3 1aa");
  } else {
    TemplatedKernel kernelBuilder(THClState_getCl(state));

    kernelBuilder
      .set("init", init)
      .set("x_threads", x_threads)
      .set("y_threads", y_threads)
      .set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS)
      .set("pair_operator2", binary_op->pair_operator2())
    ;

    kernel = kernelBuilder.buildKernel(uniqueName, "THClTensorMathTransformReduce.cl", THClTensorMathTransformReduce_getKernelTemplate(), "THClTensor_kernel_transformReduceInnermostDimIndex");
  }

  THClKernels k(state, kernel);
  k.out( tgt1 );
  k.out( tgt2 );
  k.in( src );
  k.in((int)num_rows);
  k.in((int)row_size);
  k.run(grid, threads);
}

void THClTensor_reduceDimIndex(THClState *state, THClTensor *tgt1_, THClTensor *tgt2_, THClTensor *src,
                              long dimension, float init,
                                     HasPairOperator2 *binary_op)
{
  THArgCheck(dimension >= 0 && dimension < THClTensor_nDimension(state, src), 3, "dimension out of range");

  THLongStorage *dim = THClTensor_newSizeOf(state, src);
  THLongStorage_set(dim, dimension, 1);
  THClTensor_resize(state, tgt1_, dim, NULL);
  THClTensor_resize(state, tgt2_, dim, NULL);
  THLongStorage_free(dim);

  THClTensor *tgt1 = THClTensor_newContiguous(state, tgt1_);
  THClTensor *tgt2 = THClTensor_newContiguous(state, tgt2_);
  src = THClTensor_newContiguous(state, src);

  if(dimension == THClTensor_nDimension(state, src)-1) {
    THClTensor_transformReduceInnermostDimIndex(state, tgt1, tgt2, src, init, binary_op);
  } else {
    THClTensor_transformReduceOuterDimIndex(state, tgt1, tgt2, src, dimension, init, binary_op);
  }

  THClTensor_free(state, src);
  THClTensor_freeCopyTo(state, tgt1, tgt1_);
  THClTensor_freeCopyTo(state, tgt2, tgt2_);
}

void THClTensor_max(THClState *state, THClTensor *values, THClTensor *indices, THClTensor *src, long dimension)
{
  THAssert(THClTensor_checkGPU(state, 3, values, indices, src));
  maxvalue_functor reduceOp;
  const float minfloat32 = -3.402823466e+38f;
  return THClTensor_reduceDimIndex(state, values, indices, src, dimension, minfloat32,
                                 &reduceOp);
}

void THClTensor_min(THClState *state, THClTensor *values, THClTensor *indices, THClTensor *src, long dimension)
{
  THAssert(THClTensor_checkGPU(state, 3, values, indices, src));
  minvalue_functor reduceOp;
  const float maxfloat32 = 3.402823466e+38f;
  return THClTensor_reduceDimIndex(state, values, indices, src, dimension, maxfloat32,
                                 &reduceOp);
}

std::string THClTensorMathTransformReduce_getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClTensorMathTransformReduce.cl" )
  // ]]]
  // generated using cog, from THClTensorMathTransformReduce.cl:
  const char * kernelSource =  
  "// from lib/THC/THCTensorMathTransformReduce.cu:\n" 
  "\n" 
  "typedef struct Pair {\n" 
  "   float first;\n" 
  "   float second;\n" 
  "} Pair;\n" 
  "\n" 
  "Pair binary_op( Pair a, Pair b ) {\n" 
  "    {{pair_operator2}};\n" 
  "}\n" 
  "\n" 
  "/* A set of reduction kernels that take in binary ops on thrust pairs (of value, index).\n" 
  "   These are useful when you not only have to do a reduction, but you might have\n" 
  "   to preserve the location of contention (for example min/max operations).\n" 
  "   The structure of the kernels follows the structure of the reduction kernels.\n" 
  "*/\n" 
  "kernel void THClTensor_kernel_transformReduceOuterDimIndex(global float *tgt1_data,\n" 
  "                                                          int tgt1_offset,\n" 
  "                                                          global float *tgt2_data,\n" 
  "                                                          int tgt2_offset,\n" 
  "                                                             global float *src_data,\n" 
  "                                                           int src_offset,\n" 
  "                                                             int num_orows,\n" 
  "                                                             int num_irows,\n" 
  "                                                             int row_size)\n" 
  "{\n" 
  "  for (int orow = get_group_id(0); orow < num_orows; orow += get_num_groups(0)) {\n" 
  "    for (int irow = get_group_id(1) * get_local_size(0) + get_local_id(0); irow < num_irows; irow += get_num_groups(1) * get_local_size(0)) {\n" 
  "      global float *src = src_data + src_offset + orow * row_size * num_irows + irow;\n" 
  "      Pair acc = {.first={{init}}, .second=-1};\n" 
  "      for (int col = 0; col < row_size; ++col) {\n" 
  "        Pair lhs = {*src, col+1};\n" 
  "        acc = binary_op( lhs, acc);\n" 
  "//         acc = binary_op(thrust::make_pair(*src, col+1), acc); // i+1 for 1-indexing\n" 
  "        src += num_irows;\n" 
  "      }\n" 
  "      tgt1_data[tgt1_offset + orow * num_irows + irow] = acc.first;\n" 
  "      tgt2_data[tgt2_offset + orow * num_irows + irow] = acc.second;\n" 
  "    }\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "/* Reduce the innermost dimension of a tensor (on thrust::pair functors which are (value, index))\n" 
  " *\n" 
  " * For an n-d tensor (n <= 4) where the reduction is along the innermost dimension:\n" 
  " *\n" 
  " * - block.x is the innermost dimension, i.e. dimension 0;\n" 
  " * - block.y and grid.y make up dimension 1; and\n" 
  " * - grid.x and grid z are the remaining two outer dimensions (if any)\n" 
  " *\n" 
  " * Reduction along other dimensions is handled in a separate kernel.\n" 
  " */\n" 
  "kernel void THClTensor_kernel_transformReduceInnermostDimIndex(\n" 
  "  global float *tgt1_data, int tgt1_offset, global float* tgt2_data, int tgt2_offset,\n" 
  "  global float *src_data, int src_offset,\n" 
  "  int num_rows, int row_size )\n" 
  "{\n" 
  "  local float sbuf[{{y_threads}}][{{x_threads}}];\n" 
  "  local float ibuf[{{y_threads}}][{{x_threads}}];\n" 
  "\n" 
  "  for (int block_row = get_group_id(0) * get_local_size(1); block_row < num_rows; block_row += get_local_size(1) * get_num_groups(0)) {\n" 
  "    int row = block_row + get_local_id(1);\n" 
  "//     thrust::pair<float,float> acc = init;\n" 
  "    Pair acc = { .first={{init}}, .second=-1 };\n" 
  "    if (row < num_rows) {\n" 
  "      global float *src = src_data + src_offset + row * row_size;\n" 
  "      // Sequential reduction within a thread.\n" 
  "      for (int col = get_local_id(0); col < row_size; col += get_local_size(0)) {\n" 
  "        Pair lhs = {src[col], col+1};\n" 
  "        acc = binary_op(lhs, acc);\n" 
  "      }\n" 
  "    }\n" 
  "\n" 
  "    sbuf[get_local_id(1)][get_local_id(0)] = acc.first;\n" 
  "    ibuf[get_local_id(1)][get_local_id(0)] = acc.second;\n" 
  "\n" 
  "    // Reduce intermediate values to single value.\n" 
  "    local float* sline = &sbuf[get_local_id(1)][0];\n" 
  "    local float* iline = &ibuf[get_local_id(1)][0];\n" 
  "    for (int s = 8; s > 0; s >>= 1) {\n" 
  "      if (row < num_rows && get_local_id(0) < s) {\n" 
  "        Pair arg1 = {.first=sline[get_local_id(0)], .second=iline[get_local_id(0)]};\n" 
  "        Pair arg2 = {.first=sline[get_local_id(0) + s], .second=iline[get_local_id(0) + s]};\n" 
  "        Pair res = binary_op(arg1, arg2);\n" 
  "        sline[get_local_id(0)] = res.first;\n" 
  "        iline[get_local_id(0)] = res.second;\n" 
  "      }\n" 
  "      barrier(CLK_LOCAL_MEM_FENCE);\n" 
  "    }\n" 
  "\n" 
  "    if (row < num_rows && get_local_id(0) == 0) {\n" 
  "      tgt1_data[row + tgt1_offset] = sline[0];\n" 
  "      tgt2_data[row + tgt2_offset] = iline[0];\n" 
  "    }\n" 
  "    barrier(CLK_LOCAL_MEM_FENCE);\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}



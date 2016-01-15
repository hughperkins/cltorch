// from lib/THC/THCTensorMathScan.cu:

#include "THClTensorMath.h"
#include "THClGeneral.h"
#include "THClBlas.h"
#include "THClTensorCopy.h"
//#include "THClTensorRandom.h"
#include "THClApply.h"
#include "THClReduce.h"
#include "THClTensorMathPointwise.h"
#include "THClDeviceUtils.h"
#include "THClKernels.h"

#include <string>
using namespace std;


std::string THClTensorMathScan_getKernelTemplate();

inline long getBlockSize(THClState *state) {
  int blockSize = 1024;
  int maxWorkgroupSize = ((easycl::DeviceInfo *)state->deviceInfoByDevice[state->currentDevice])->maxWorkGroupSize;
  if( blockSize > maxWorkgroupSize ) {
    blockSize = maxWorkgroupSize;
  }
  return blockSize;
}

void THClTensor_scanOuterDim(THClState *state, THClTensor *tgt, THClTensor *src, long dimension,
                                        float init, HasOperator3 *binary_op)
{
  StatefulTimer::timeCheck("THClTensorMathScan_scanOuterDim START");
  unsigned ndim = THClTensor_nDimension(state, src);
  // Treat all outer dimensions (i.e. dim < dimension) as one.
  unsigned num_orows = 1;
  for (unsigned dim = 0; dim < dimension; dim++) {
    num_orows *= THClTensor_size(state, src, dim);
  }
  unsigned row_size = THClTensor_size(state, src, dimension);
  // Treat all inner dimensions (i.e. dim > dimension) as one.
  unsigned num_irows = 1;
  for (unsigned dim = dimension + 1; dim < ndim; dim++) {
    num_irows *= THClTensor_size(state, src, dim);
  }

  dim3 threads(THCl_min(getBlockSize(state) / 2, num_irows));
  unsigned maxGridDim = getBlockSize(state);
  dim3 grid(THCl_min(maxGridDim, num_orows), THCl_min(maxGridDim, THClCeilDiv(num_irows, threads.x())));

  std::string uniqueName = "THClTensorMathScan_scanOuterDim_" + binary_op->operator3();
  EasyCL *cl = src->storage->cl;
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("Apply3 1aa");
  } else {
    TemplatedKernel kernelBuilder(cl);
    kernelBuilder.set("num_threads_x", 16); // dont need for this kernel, but do need so whole file
    kernelBuilder.set("num_threads_y", 32); // compiles ok
    kernelBuilder.set("operator3", binary_op->operator3());

    kernel = kernelBuilder.buildKernel(uniqueName, "THClTensorMathScan.cl",
      THClTensorMathScan_getKernelTemplate(), "THClTensor_kernel_scanOuterDim");
  }

  THClKernels k(state, kernel);
  k.out(tgt);
  k.inout(src);
  k.in((int)num_orows);
  k.in((int)num_irows);
  k.in((int)row_size);
  k.in(init);

  k.run(grid, threads);
  if(state->addFinish) cl->finish();  
  StatefulTimer::timeCheck("THClTensorMathScan_scanOuterDim END");
}


void THClTensor_scanInnermostDim(THClState *state, THClTensor *tgt, THClTensor *src, float init, HasOperator3 *binary_op)
{
  StatefulTimer::timeCheck("THClTensorMathScan_scanInnermostDim START");
  unsigned ndim = THClTensor_nDimension(state, src);
  // Treat all outer dimensions as a single dimension.
  unsigned num_rows = 1;
  for (unsigned dim = 0; dim < ndim - 1; dim++) {
    num_rows *= THClTensor_size(state, src, dim);
  }
  unsigned row_size = THClTensor_size(state, src, ndim - 1);

  int x_threads = 16;
  int y_threads = 32;
  if( getBlockSize(state) < x_threads * y_threads ) {
    y_threads = getBlockSize(state) / x_threads;
  }
  dim3 threads(x_threads, y_threads);
  dim3 grid(THCl_min(getBlockSize(state), THClCeilDiv(num_rows, threads.y())));

  std::string uniqueName = "THClTensorMathScan_scanInnermostDim_" + binary_op->operator3();
  EasyCL *cl = src->storage->cl;
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("Apply3 1aa");
  } else {
    TemplatedKernel kernelBuilder(cl);
    kernelBuilder.set("num_threads_x", x_threads);
    kernelBuilder.set("num_threads_y", y_threads);
    kernelBuilder.set("operator3", binary_op->operator3());

    kernel = kernelBuilder.buildKernel(uniqueName, "THClTensorMathScan.cl",
      THClTensorMathScan_getKernelTemplate(), "THClTensor_kernel_scanInnermostDim");
  }

  THClKernels k(state, kernel);
  k.out(tgt);
  k.inout(src);
  k.in((int)num_rows);
  k.in((int)row_size);
  k.in(init);

  k.run(grid, threads);
  if(state->addFinish) cl->finish();  
  StatefulTimer::timeCheck("THClTensorMathScan_scanInnermostDim END");
}

void THClTensor_scanDim(THClState *state, THClTensor *self_, THClTensor *src, long dimension, float init, HasOperator3 *binary_op)
{
  THClTensor_resizeAs(state, self_, src);

  THClTensor *self = THClTensor_newContiguous(state, self_);
  src = THClTensor_newContiguous(state, src);

  if (dimension == THClTensor_nDimension(state, src) - 1) {
    THClTensor_scanInnermostDim(state, self, src, init, binary_op);
  } else {
    THClTensor_scanOuterDim(state, self, src, dimension, init, binary_op);
  }

  THClTensor_free(state, src);
  THClTensor_freeCopyTo(state, self, self_);
}

void THClTensor_cumsum(THClState *state, THClTensor *self, THClTensor *src, long dimension)
{
  TensorAddOp cumOp;
  THAssert(THClTensor_checkGPU(state, 2, self, src));
  THClTensor_scanDim(state, self, src, dimension, 0.0f, &cumOp);
}

void THClTensor_cumprod(THClState *state, THClTensor *self, THClTensor *src, long dimension)
{
  TensorMulOp cumOp;
  THAssert(THClTensor_checkGPU(state, 2, self, src));
  THClTensor_scanDim(state, self, src, dimension, 1.0f, &cumOp);
}

std::string THClTensorMathScan_getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClTensorMathScan.cl" )
  // ]]]
  // generated using cog, from THClTensorMathScan.cl:
  const char * kernelSource =  
  "// from lib/THC/THCTensorMathScan.cu:\n" 
  "\n" 
  "/* Perform an inclusive scan along an outer dimension of a tensor.\n" 
  " *\n" 
  " * - num_orows is the size of the flattened outer dimensions;\n" 
  " * - num_irows is the size of the flattened inner dimensions;\n" 
  " * - row_size is the size of the dimension along which to compute the variance;\n" 
  " *\n" 
  " * The dimensions to the outside and inside of the specified dimension are considered as flattened.\n" 
  " * Thread blocks with the same get_group_id(1) process an \"outer row\" (i.e. an element of the flattened\n" 
  " * outer dimensions, which contains several \"inner rows\").\n" 
  " * Each thread processes a single inner row at a time.\n" 
  " */\n" 
  "\n" 
  "inline float binary_op(float _in1, float _in2) {\n" 
  "  // hope the compiler can handle this :-P\n" 
  "  float _out;\n" 
  "  float *out = &_out;\n" 
  "  float *in1 = &_in1;\n" 
  "  float *in2 = &_in2;\n" 
  "  *out = 10 * (*in2) * (*in1);\n" 
  "  {{operator3}};\n" 
  " // *out = (*in1) * (*in2);\n" 
  "  return _out;\n" 
  "}\n" 
  "\n" 
  "kernel void THClTensor_kernel_scanOuterDim(\n" 
  "  global float *tgt_data, int tgt_offset,\n" 
  "  global float *src_data, int src_offset,\n" 
  "  int num_orows, int num_irows, int row_size,\n" 
  "  float init)\n" 
  "{\n" 
  "  for (unsigned orow = get_group_id(0); (int)orow < num_orows; orow += get_num_groups(0)) {\n" 
  "    for (unsigned irow = get_group_id(1) * get_local_size(0) + get_local_id(0); (int)irow < num_irows; irow += get_num_groups(1) * get_local_size(0)) {\n" 
  "      global float *src = src_data + src_offset + orow * row_size * num_irows + irow;\n" 
  "      global float *tgt = tgt_data + tgt_offset + orow * row_size * num_irows + irow;\n" 
  "      float acc = init;\n" 
  "\n" 
  "      for (unsigned col = 0; (int)col < row_size; ++col) {\n" 
  "        acc = binary_op(acc, *src);\n" 
  "//        binary_op(&acc, &acc, src);\n" 
  "        *tgt = acc;\n" 
  "\n" 
  "        src += num_irows;\n" 
  "        tgt += num_irows;\n" 
  "      }\n" 
  "    }\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "/* Perform an inclusive scan along the innermost dimension of a tensor.\n" 
  " *\n" 
  " * - num_rows is the size of the flattened outer dimensions;\n" 
  " * - row_size is the size of the innermost dimension;\n" 
  " *\n" 
  " * The outer dimensions of the tensor are considered as a single dimension, i.e. the tensor is\n" 
  " * considered as having 'num_rows' rows of size 'row_size'.\n" 
  " * Each thread block processes one or more sets of contiguous rows (processing multiple rows\n" 
  " * per thread block is quicker than processing a single row, especially for short rows).\n" 
  " */\n" 
  "//template<int num_threads_x, int num_threads_y, class BinaryFunction>\n" 
  "kernel void THClTensor_kernel_scanInnermostDim(\n" 
  "  global float *tgt_data, int tgt_offset,\n" 
  "  global float *src_data, int src_offset,\n" 
  "  int num_rows, int row_size,\n" 
  "  float init)\n" 
  "{\n" 
  "  local float sbuf[{{num_threads_y}}][2 * {{num_threads_x}}];\n" 
  "\n" 
  "  local float* row_buf = sbuf[get_local_id(1)];\n" 
  "\n" 
  "  for (int block_row = get_group_id(0) * get_local_size(1);\n" 
  "       block_row < num_rows;\n" 
  "       block_row += get_local_size(1) * get_num_groups(0)) {\n" 
  "    int row = block_row + get_local_id(1);\n" 
  "    float block_total = init;\n" 
  "\n" 
  "    global float *row_src = src_data + src_offset + row * row_size;\n" 
  "    global float *row_tgt = tgt_data + tgt_offset + row * row_size;\n" 
  "\n" 
  "    // Perform scan on one block at a time, keeping track of the total value of\n" 
  "    // all blocks processed so far.\n" 
  "    for (int block_col = 0; block_col < (int)row_size; block_col += 2 * {{num_threads_x}}) {\n" 
  "      // Load data into shared memory (two values per thread).\n" 
  "      int col1 = block_col + get_local_id(0);\n" 
  "      int col2 = block_col + {{num_threads_x}} + get_local_id(0);\n" 
  "      if (row < num_rows) {\n" 
  "        if (col1 < row_size) {\n" 
  "          row_buf[get_local_id(0)] = row_src[col1];\n" 
  "        } else {\n" 
  "          row_buf[get_local_id(0)] = init;\n" 
  "        }\n" 
  "\n" 
  "        if (col2 < row_size) {\n" 
  "          row_buf[{{num_threads_x}} + get_local_id(0)] = row_src[col2];\n" 
  "        } else {\n" 
  "          row_buf[{{num_threads_x}} + get_local_id(0)] = init;\n" 
  "        }\n" 
  "\n" 
  "        // Add the total value of all previous blocks to the first value of this block.\n" 
  "        if (get_local_id(0) == 0) {\n" 
  "          row_buf[0] = binary_op(row_buf[0], block_total);\n" 
  "//          binary_op(row_buf, row_buf, &block_total);\n" 
  "        }\n" 
  "      }\n" 
  "      barrier(CLK_LOCAL_MEM_FENCE);\n" 
  "\n" 
  "      // Parallel reduction (up-sweep).\n" 
  "      for (int s = {{num_threads_x}}, d = 1; s >= 1; s >>= 1, d <<= 1) {\n" 
  "        if (row < num_rows && (int)get_local_id(0) < s) {\n" 
  "          int offset = (2 * get_local_id(0) + 1) * d - 1;\n" 
  "          row_buf[offset + d] = binary_op(row_buf[offset], row_buf[offset + d]);\n" 
  "//          binary_op(row_bufer + offset + d, row_buf + offset, row_buf + offset + d);\n" 
  "        }\n" 
  "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
  "      }\n" 
  "\n" 
  "      // Down-sweep.\n" 
  "      for (int s = 2, d = {{num_threads_x}} / 2; d >= 1; s <<= 1, d >>= 1) {\n" 
  "        if (row < num_rows && (int)get_local_id(0) < s - 1) {\n" 
  "          int offset = 2 * (get_local_id(0) + 1) * d - 1;\n" 
  "          row_buf[offset + d] = binary_op(row_buf[offset], row_buf[offset + d]);\n" 
  "//          binary_op(row_buff + offset + d, row_buf + offset, row_buf + offset + d);\n" 
  "        }\n" 
  "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
  "      }\n" 
  "\n" 
  "\n" 
  "      // Write back to output.\n" 
  "      if (row < num_rows) {\n" 
  "        if (col1 < row_size) row_tgt[col1] = row_buf[get_local_id(0)];\n" 
  "        if (col2 < row_size) row_tgt[col2] = row_buf[{{num_threads_x}} + get_local_id(0)];\n" 
  "      }\n" 
  "      block_total = row_buf[2 * {{num_threads_x}} - 1];\n" 
  "      barrier(CLK_LOCAL_MEM_FENCE);\n" 
  "\n" 
  "    }\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}


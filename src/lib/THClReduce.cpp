#include <iostream>

#include "THClReduce.h"
#include "THClTypeParseTraits.h"

using namespace std;

#define THCL_NONCONTIG_REDUCE_BLOCK_SIZE 32 * 16

static int getNonContigReduceBlockSize(THClState *state) {
  int blockSize = 1024;
  int maxWorkgroupSize = ((easycl::DeviceInfo *)state->deviceInfoByDevice[state->currentDevice])->maxWorkGroupSize;
  if( blockSize > maxWorkgroupSize ) {
    blockSize = maxWorkgroupSize;
  }
  return blockSize;
}

static dim3 getNoncontigReduceBlock(THClState *state) {
  return dim3(getNonContigReduceBlockSize(state));
}

static dim3 getContigReduceBlock(THClState *state, int64 numSlices, int64 reductionSize) {
  // If the number of slices is low but the reduction dimension size
  // is high, then we should increase block size for greater parallelism.
  // Aim for at least 32 warps per SM (assume 15 SMs; don't bother
  // inquiring the real number for now).
  int maxWarps = 4; // better occupancy if many blocks are around
  // For numSlices > 15 * 8, there are > 32 warps active per SM.
  if (numSlices < 15 * 8) {
    maxWarps = 8;
    if (numSlices < 15 * 4) {
      maxWarps = 16;
      if (numSlices < 15 * 2) {
        maxWarps = 32;
      }
    }
  }

  // Scale up block size based on the reduction dimension size
  int64 warpsInReductionSize = THClCeilDiv(reductionSize, 32ll);
//  int64 warpsInReductionSize = DIVUP(reductionSize, 32L);
  int numWarps =
    warpsInReductionSize > (int64) maxWarps ? maxWarps : (int) warpsInReductionSize;
//    warpsInReductionSize > (int64) maxWarps ? maxWarps : (int) warpsInReductionSize;

  int targetSize = numWarps * 32;
  // but is this greater than maxWorkgroupSize?  We should check :-)
  int maxWorkgroupSize = ((easycl::DeviceInfo *)state->deviceInfoByDevice[state->currentDevice])->maxWorkGroupSize;
  if(targetSize > maxWorkgroupSize) {
    targetSize = maxWorkgroupSize;
  }

  return dim3(targetSize);
}

static bool getNoncontigReduceGrid(THClState *state, int64 elements, dim3& grid) {
  // One output point per thread
  return THCL_getGridFromTiles(THClCeilDiv(elements, (int64) getNonContigReduceBlockSize(state)), grid);
//  return THCL_getGridFromTiles(DIVUP(elements, THCL_NONCONTIG_REDUCE_BLOCK_SIZE), grid);
}

static bool getContigReduceGrid(int64 elements, dim3& grid) {
  // One output point per block
  return THCL_getGridFromTiles(elements, grid);
}

template<typename IndexType>
void kernelLaunch_THClTensor_reduceNoncontigDim(
  THClState *state,
  dim3 &grid,
  dim3 &block,
  int ADims,
  int BDims,
  TensorInfo<IndexType> out,
  TensorInfo<IndexType> in,
  IndexType reductionStride,
  IndexType reductionSize,
  IndexType totalSlices,
  float init,
  HasOperator2 const*modifyOp,
  HasOperator3 const*reduceOp) {

//  cl->finish();
  StatefulTimer::timeCheck("Reduce-noncontig START");
  std::string uniqueName = "THClTensor_reduceNoncontigDim_" + easycl::toString(ADims) + "_" + easycl::toString(BDims) + "_" +
    TypeParseTraits<IndexType>::name + "_" + modifyOp->operator2() + "_" + reduceOp->operator3();
  EasyCL *cl = in.wrapper->getCl();
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("Apply3 1aa");
  } else {
    // launch kernel here....
    TemplatedKernel kernelBuilder(cl);

    std::set<int> dims_set;
    if(ADims >= 0) {
      dims_set.insert(ADims);
    }
    if(BDims >= 0) {
      dims_set.insert(BDims);
    }
    std::vector<int> dims;
    for( std::set<int>::iterator it = dims_set.begin(); it != dims_set.end(); it++ ) {
      dims.push_back(*it);
    }

    std::string indexType = TypeParseTraits<IndexType>::name;
    kernelBuilder
      .set("include_THClReduceApplyUtils", THClReduceApplyUtils_getKernelTemplate())
      .set("dims", dims)
      .set("dim1", ADims)
      .set("dim2", BDims)
      .set("defreduceblock", 1)
      .set("WarpSize", 32) // probably can do like 'if nvidia 32 else 64' ?
      .set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS)
      .set("IndexType", indexType)
      .set("modify_operation", modifyOp->operator2())
      .set("reduce_operation", reduceOp->operator3())
    ;

    kernel = kernelBuilder.buildKernel(uniqueName, "THClReduce.cl", THClReduce_getKernelSource(), "THClTensor_reduceNoncontigDim");
  }

  THClKernels k(state, kernel);
  k.out(out);
  k.in(in);
  k.in((int)reductionStride);
  k.in((int)reductionSize);
  k.in((int)totalSlices);
  k.in(init);

  k.run(grid, block);

  if(state->addFinish) cl->finish();  
  StatefulTimer::timeCheck("Reduce-noncontig END");
}

template<typename IndexType>
void kernelLaunch_THClTensor_reduceContigDim(
  THClState *state,
  dim3 &grid,
  dim3 &block,
  size_t smemSize,
  int ADims,
  int BDims,
  TensorInfo<IndexType> out,
  TensorInfo<IndexType> in,
  IndexType reductionSize,
  IndexType totalSlices,
  float init,
  HasOperator2 const*modifyOp,
  HasOperator3 const*reduceOp) {
  StatefulTimer::timeCheck("Reduce-contig START");
  std::string uniqueName = "THClTensor_reduceContigDim_" + easycl::toString(ADims) + "_" + easycl::toString(BDims) + "_" +
    TypeParseTraits<IndexType>::name + "_" + modifyOp->operator2() + "_" + reduceOp->operator3();
  EasyCL *cl = in.wrapper->getCl();
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("Apply3 1aa");
  } else {

    // launch kernel here....

    TemplatedKernel kernelBuilder(cl);

    std::set<int> dims_set;
    if(ADims >= 0) {
      dims_set.insert(ADims);
    }
    if(BDims >= 0) {
      dims_set.insert(BDims);
    }
    std::vector<int> dims;
    for( std::set<int>::iterator it = dims_set.begin(); it != dims_set.end(); it++ ) {
      dims.push_back(*it);
    }

    std::string indexType = TypeParseTraits<IndexType>::name;
    kernelBuilder
      .set("include_THClReduceApplyUtils", THClReduceApplyUtils_getKernelTemplate())
      .set("dims", dims)
      .set("dim1", ADims)
      .set("defreduceblock", 1)
      .set("dim2", BDims)
      .set("WarpSize", 32) // probably can do like 'if nvidia 32 else 64' ?
      .set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS)
  //    .set("IndexType", indexType)
      .set("IndexType", "int")
      .set("modify_operation", modifyOp->operator2())
      .set("reduce_operation", reduceOp->operator3())
    ;

    kernel = kernelBuilder.buildKernel(uniqueName, "THClReduce.cl", THClReduce_getKernelSource(), "THClTensor_reduceContigDim");
  }

  THClKernels k(state, kernel);
  k.out(out);
  k.in(in);
  k.in((int)reductionSize);
  k.in((int)totalSlices);
  k.in(init);
  k.localFloats(smemSize / sizeof(float));

  k.run(grid, block);

  if(state->addFinish) cl->finish();
  StatefulTimer::timeCheck("Reduce-contig END");
}

// Performs a reduction out[..., 0, ...] = reduce_i(modify(in[..., i, ...])) for
// all in where i and the out's 0 are indexed at dimension `dim`
bool THClTensor_reduceDim(THClState* state,
                            THClTensor* out,
                            THClTensor* in,
                            float init,
                            const HasOperator2 *modifyOp,
                            const HasOperator3 *reduceOp,
                            int dim) {
  int64 inElements = THClTensor_nElement(state, in);

  int64 reductionSize = THClTensor_size(state, in, dim);
  int64 reductionStride = THClTensor_stride(state, in, dim);
  int64 outElements = inElements / reductionSize;

  if (THClTensor_nDimension(state, out) > MAX_CLTORCH_DIMS ||
      THClTensor_nDimension(state, in) > MAX_CLTORCH_DIMS) {
    return false;
  }

  if (THClTensor_nDimension(state, in) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  // Is the reduction dimension contiguous? If so, then we can use a
  // shared memory reduction kernel to increase performance.
  bool contigReduction = (reductionStride == 1);

  dim3 block;
  dim3 grid;
  int smemSize = 0; // contiguous reduction uses smem
  if (contigReduction) {
    if (!getContigReduceGrid(outElements, grid)) {
      return false;
    }
    block = getContigReduceBlock(state, outElements, reductionSize);
    smemSize = sizeof(float) * block.x();
  } else {
    if (!getNoncontigReduceGrid(state, outElements, grid)) {
      return false;
    }
    block = getNoncontigReduceBlock(state);
  }

  // Resize out to correspond to the reduced size
  THLongStorage* sizes = THClTensor_newSizeOf(state, in);
  THLongStorage_set(sizes, dim, 1);
  THClTensor_resize(state, out, sizes, NULL);
  THLongStorage_free(sizes);

  if (THCL_canUse32BitIndexMath(state, out) &&
      THCL_canUse32BitIndexMath(state, in)) {
    TensorInfo<uint32> outInfo(state, out);
    TensorInfo<uint32> inInfo(state, in, dim);
    int OUT = outInfo.dims;
    int IN = inInfo.dims;
    if(outInfo.isContiguous()) OUT = -2;
    if(inInfo.isContiguous()) IN = -2;

    if (contigReduction) {
      kernelLaunch_THClTensor_reduceContigDim<uint32> (
          state, grid, block, smemSize, OUT, IN, outInfo, inInfo, (uint32) reductionSize,
          (uint32) outElements, init, modifyOp, reduceOp);
    } else {
       kernelLaunch_THClTensor_reduceNoncontigDim<uint32> (
          state, grid, block, OUT, IN, outInfo, inInfo, (uint32) reductionStride, (uint32) reductionSize,
          (uint32) outElements, init, modifyOp, reduceOp);
    }
  } else {
    TensorInfo<uint64> outInfo(state, out);
    TensorInfo<uint64> inInfo(state, in, dim);

    int OUT = outInfo.dims;
    int IN = inInfo.dims;
    if(outInfo.isContiguous()) OUT = -2;
    if(inInfo.isContiguous()) IN = -2;

    if (contigReduction) {
      kernelLaunch_THClTensor_reduceContigDim<uint64> (
          state, grid, block, smemSize, OUT, IN, outInfo, inInfo, (uint64) reductionSize,
          (uint64) outElements, init, modifyOp, reduceOp);
    } else {
       kernelLaunch_THClTensor_reduceNoncontigDim<uint64> (
          state, grid, block, OUT, IN, outInfo, inInfo, (uint64) reductionStride, (uint64) reductionSize,
          (uint64) outElements, init, modifyOp, reduceOp);
    }
  }
  return true;
}

std::string THClReduce_getKernelSource() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClReduce.cl" )
  // ]]]
  // generated using cog, from THClReduce.cl:
  const char * kernelSource =  
  "// Threads per thread block\n" 
  "#define THCL_NONCONTIG_REDUCE_BLOCK_SIZE 32 * 16\n" 
  "\n" 
  "static inline float modifyOp(float _in1) {\n" 
  "  float _out;\n" 
  "  float *in1 = &_in1;\n" 
  "  float *out = &_out;\n" 
  "  {{modify_operation}};\n" 
  "  return _out;\n" 
  "}\n" 
  "\n" 
  "static inline float reduceOp(float _in1, float _in2) {\n" 
  "  // I guess the compiler can sort this stuff out :-P\n" 
  "  float _out;\n" 
  "  float *in1 = &_in1;\n" 
  "  float *in2 = &_in2;\n" 
  "  float *out = &_out;\n" 
  "  {{reduce_operation}};\n" 
  "  return _out;\n" 
  "}\n" 
  "\n" 
  "{{include_THClReduceApplyUtils}}\n" 
  "\n" 
  "static inline {{IndexType}} getReduceNoncontigDimSliceIndex() {\n" 
  "  // Each thread handles one slice\n" 
  "  return getLinearBlockId() * THCL_NONCONTIG_REDUCE_BLOCK_SIZE + /*threadIdx.x*/ get_local_id(0);\n" 
  "}\n" 
  "\n" 
  "// Kernel that handles an entire reduction of a slice of a tensor per each thread\n" 
  "kernel void\n" 
  "THClTensor_reduceNoncontigDim(global TensorInfoCl *out_info,\n" 
  "                              global float *out_data,\n" 
  "                              global TensorInfoCl *in_info,\n" 
  "                              global float *in_data,\n" 
  "                              int reductionStride,\n" 
  "                              int reductionSize,\n" 
  "                              int totalSlices,\n" 
  "                              float init) {\n" 
  "  const {{IndexType}} sliceIndex = getReduceNoncontigDimSliceIndex();\n" 
  "\n" 
  "  if ((int)sliceIndex >= totalSlices) {\n" 
  "    return;\n" 
  "  }\n" 
  "\n" 
  "  // Each thread picks a point in `out` and `in` for which it is\n" 
  "  // producing the reduction\n" 
  "  const {{IndexType}} outOffset =\n" 
  "    IndexToOffset_{{1000 + dim1}}_get(sliceIndex, &out_info[0]);\n" 
  "  const {{IndexType}} inBaseOffset =\n" 
  "    IndexToOffset_{{1000 + dim2}}_get(sliceIndex, &in_info[0]);\n" 
  "\n" 
  "  // For each point in reductionSize, reduce into `r`\n" 
  "  {{IndexType}} inOffset = inBaseOffset;\n" 
  "  float r = init;\n" 
  "\n" 
  "  for ({{IndexType}} i = 0; (int)i < reductionSize; ++i) {\n" 
  "    r = reduceOp(r, modifyOp(in_data[inOffset]));\n" 
  "    inOffset += reductionStride;\n" 
  "  }\n" 
  "\n" 
  "  // Write out reduced value\n" 
  "  out_data[outOffset] = r;\n" 
  "}\n" 
  "\n" 
  "static inline {{IndexType}} getReduceContigDimSliceIndex() {\n" 
  "  // Each block handles one slice\n" 
  "  return getLinearBlockId();\n" 
  "}\n" 
  "\n" 
  "// Kernel that handles an entire reduction of a slice of a tensor per\n" 
  "// each block\n" 
  "kernel void\n" 
  "THClTensor_reduceContigDim(global TensorInfoCl *out_info,\n" 
  "                           global float *out_data,\n" 
  "                           global TensorInfoCl *in_info,\n" 
  "                           global float *in_data,\n" 
  "                           int reductionSize,\n" 
  "                           int totalSlices,\n" 
  "                           float init,\n" 
  "                           local float *smem) {\n" 
  "  const {{IndexType}} sliceIndex = getReduceContigDimSliceIndex();\n" 
  "\n" 
  "  if ((int)sliceIndex >= totalSlices) {\n" 
  "    return;\n" 
  "  }\n" 
  "\n" 
  "  // Get the offset in `out` for the reduction\n" 
  "  const {{IndexType}} outOffset =\n" 
  "    IndexToOffset_{{1000 + dim1}}_get(sliceIndex, &out_info[0]);\n" 
  "\n" 
  "  // Get the base offset in `in` for this block's reduction\n" 
  "  const {{IndexType}} inBaseOffset =\n" 
  "    IndexToOffset_{{1000 + dim2}}_get(sliceIndex, &in_info[0]);\n" 
  "\n" 
  "  // Each thread in the block will reduce some subset of elements in\n" 
  "  // the slice. The elements are guaranteed contiguous starting at\n" 
  "  // `inBaseOffset`.\n" 
  "  float r = init;\n" 
  "  for ({{IndexType}} i = /*threadIdx.x*/ get_local_id(0); (int)i < reductionSize; i += /*blockDim.x*/ get_local_size(0)) {\n" 
  "    r = reduceOp(r, modifyOp(in_data[inBaseOffset + i]));\n" 
  "  }\n" 
  "\n" 
  "  // Reduce within the block\n" 
  "//  extern __shared__ float smem[];\n" 
  "  r = reduceBlock(smem, /*blockDim.x*/ get_local_size(0), r, init);\n" 
  "\n" 
  "  if (/*threadIdx.x*/ get_local_id(0) == 0) {\n" 
  "    // Write out reduced value\n" 
  "    out_data[outOffset] = r;\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}



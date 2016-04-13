#include <string>
#include <iostream>

#include "THClReduceApplyUtils.cuh"
#include "THClSortUtils.cuh"
#include "THClTensorCopy.h"

#include "EasyCL.h"
#include "CLKernel_structs.h"
#include "util/easycl_stringhelper.h"
#include "util/StatefulTimer.h"
#include "templates/TemplatedKernel.h"

using namespace std;

static std::string THClSortUtils_getKernelTemplate();


//template< typename IndexType >
//void kernelLaunch_pointwiseApply( THClState *state, dim3 grid, dim3 block, int numTensors, int *dims, TensorInfo<IndexType> **infos, IndexType totalElements, OpBase const*op, string operationString) {

template< typename IndexType >
void kernelLaunch_bitonicSortKVInPlace(
    THClState *state,
    dim3 grid, dim3 block,
    TensorInfo<IndexType> *keyInfo,
    IndexType keySlices,
    IndexType keySliceSize,
    IndexType keySliceStride,
    TensorInfo<IndexType> *valueInfo,
    IndexType valueSliceStride,
    SortUtilsComp &comp) {
  StatefulTimer::timeCheck("bitonicSortKVInPlace START");
  std::string uniqueName = std::string("THClSortUtils_bitonicSortKVInPlace_") + comp->getOperator();
  EasyCL *cl = scratch->getCl();
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("bitonicSortKVInPlace 1aa");
  } else {
    TemplatedKernel kernelBuilder(cl);
    std::vector< int > dims;
    kernelBuilder
//      .set("include_THClDeviceUtils", THClDeviceUtils_getKernelTemplate())
      .set("include_THClReduceApplyUtils", THClReduceApplyUtils_getKernelTemplate())
//      .set("WarpSize", 32) // probably can do like 'if nvidia 32 else 64' ?
//      .set("dims", dims)
//      .set("dim1", -2)
//      .set("defreduceblock", 1)
//      .set("reduce_operation", reduceOp->operator3())
      .set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS)
      .set("IndexType", TypeParseTraits<IndexType>::name)
    ;

    kernel = kernelBuilder.buildKernel( uniqueName, "THClSortUtils.cl", getKernelTemplate(), "bitonicSortKVInPlace" );
  }

  THClKernels k(state, kernel);
  k.in(numPass1Blocks);
  k.in(init);
  k.in(scratch);
  k.out(devOut);
  k.localFloats(smemSize / sizeof(float));
  k.run(grid, block);

  if(state->addFinish) cl->finish();  
  StatefulTimer::timeCheck("ReduceAllPass2 End");
}

//template< typename IndexType >
//void kernelLaunch_THClTensor_reduceAllPass2(
//                     THClState* state,
//                     dim3 &grid, dim3 &block, size_t smemSize,
//                     int numPass1Blocks,
//                     float init,
//                     const HasOperator3 *reduceOp,
//                     CLWrapper *scratch,
//                     CLWrapper* devOut
//    ){
//  StatefulTimer::timeCheck("ReduceAllPass2 START");
//  std::string uniqueName = "THClTensor_reduceAllPass2_" + reduceOp->operator3();
//  EasyCL *cl = scratch->getCl();
//  CLKernel *kernel = 0;
//  if(cl->kernelExists(uniqueName)) {
//    kernel = cl->getKernel(uniqueName);
//    StatefulTimer::timeCheck("ReduceAllPass2 1aa");
//  } else {
//    TemplatedKernel kernelBuilder(cl);
//    std::vector< int > dims;
//    kernelBuilder
//      .set("include_THClDeviceUtils", THClDeviceUtils_getKernelTemplate())
//      .set("include_THClReduceApplyUtils", THClReduceApplyUtils_getKernelTemplate())
//      .set("WarpSize", 32) // probably can do like 'if nvidia 32 else 64' ?
//      .set("dims", dims)
//      .set("dim1", -2)
//      .set("defreduceblock", 1)
//      .set("reduce_operation", reduceOp->operator3())
//      .set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS)
//      .set("IndexType", TypeParseTraits<IndexType>::name)
//    ;

//    kernel = kernelBuilder.buildKernel( uniqueName, "THClReduceAll.cl", getKernelTemplate(), "THClTensor_reduceAllPass2" );
//  }

//  THClKernels k(state, kernel);
//  k.in(numPass1Blocks);
//  k.in(init);
//  k.in(scratch);
//  k.out(devOut);
//  k.localFloats(smemSize / sizeof(float));
//  k.run(grid, block);

//  if(state->addFinish) cl->finish();  
//  StatefulTimer::timeCheck("ReduceAllPass2 End");
//}


//  "bitonicSortKVInPlace(TensorInfo<{{IndexType}}> keys,\n" 
//  "                     {{IndexType}} keySlices,\n" 
//  "                     {{IndexType}} keySliceSize,\n" 
//  "                     {{IndexType}} keySliceStride,\n" 
//  "                     TensorInfo<{{IndexType}}> values,\n" 
//  "                     {{IndexType}} valueSliceStride) {\n" 

std::string THClSortUtils_getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClSortUtils.cl" )
  // ]]]
  // generated using cog, from THClSortUtils.cl:
  const char * kernelSource =  
  "// from lib/THC/THCSortUtils.cuh:\n" 
  "\n" 
  "// This needs the following template variables:\n" 
  "//   K              key type\n" 
  "//   V              value type\n" 
  "//   COMPARE_OP     a comparison operator, like <   or >\n" 
  "//   KeyDims        integer\n" 
  "//   ValueDims      integer\n" 
  "//   Power2SortSize  integer\n" 
  "\n" 
  "// you need to somewhere include {{THClReduceApplyUtils}} before this, with appropriate dims, to include\n" 
  "// KeyDims and ValueDims\n" 
  "\n" 
  "\n" 
  "/*__device__*/ inline void swapVars_K({{K}} *p_t1, {{K}}*p_t2) {\n" 
  "  {{K}} tmp = *p_t1;\n" 
  "  *p_t1 = *p_t2;\n" 
  "  *p_t2 = tmp;\n" 
  "}\n" 
  "\n" 
  "/*__device__*/ inline void swapVars_V({{V}} *p_t1, {{V}}*p_t2) {\n" 
  "  {{V}} tmp = *p_t1;\n" 
  "  *p_t1 = *p_t2;\n" 
  "  *p_t2 = tmp;\n" 
  "}\n" 
  "\n" 
  "/*__device__*/ inline void swapVars_bool(bool *p_t1, bool *p_t2) {\n" 
  "  bool tmp = *p_t1;\n" 
  "  *p_t1 = *p_t2;\n" 
  "  *p_t2 = tmp;\n" 
  "}\n" 
  "\n" 
  "/*__device__*/ inline void bitonicSwap({{K}}* p_kA, V*p_vA, bool*p_validA,\n" 
  "                                   {{K}}* p_kB, V*p_vB, bool*p_validB,\n" 
  "                                   bool dir) {\n" 
  "  // Invalid entries always sort to the end\n" 
  "  bool swap = (kA {{COMPARE_OP}} kB) && validA) || !validB;\n" 
  "  if (swap == dir) {\n" 
  "    swapVars_K(p_kA, p_kB);\n" 
  "    swapVars_V(p_vA, p_vB);\n" 
  "    swapVars_bool(p_validA, p_validB);\n" 
  "  }\n" 
  "};\n" 
  "\n" 
  "template <int Power2SortSize>\n" 
  "/*__device__*/ inline void bitonicSort({{K}} keys[{{Power2SortSize}}],\n" 
  "                                   {{V}} values[{{Power2SortSize}}],\n" 
  "                                   bool valid[{{Power2SortSize}}]) {\n" 
  "#pragma unroll\n" 
  "  for (unsigned int size = 2; size < {{Power2SortSize}}; size *= 2) {\n" 
  "    bool flag = ((get_local_id(0) & (size / 2)) != 0);\n" 
  "\n" 
  "#pragma unroll\n" 
  "    for (unsigned int stride = size / 2; stride > 0; stride /= 2) {\n" 
  "\n" 
  "      // Single warp per slice is completely synchronous\n" 
  "      if (Power2SortSize > 64) {\n" 
  "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
  "      }\n" 
  "\n" 
  "      unsigned int pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));\n" 
  "      bitonicSwap(\n" 
  "        &keys[pos], &values[pos], &valid[pos],\n" 
  "        &keys[pos + stride], &values[pos + stride], &valid[pos + stride],\n" 
  "        flag);\n" 
  "    }\n" 
  "  }\n" 
  "\n" 
  "#pragma unroll\n" 
  "  for (unsigned int stride = Power2SortSize / 2; stride > 0; stride /= 2) {\n" 
  "    // Single warp per slice is completely synchronous\n" 
  "    if (Power2SortSize > 64) {\n" 
  "      barrier(CLK_LOCAL_MEM_FENCE);\n" 
  "    }\n" 
  "\n" 
  "    unsigned int pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));\n" 
  "    bitonicSwap(\n" 
  "      &keys[pos], &values[pos], &valid[pos],\n" 
  "      &keys[pos + stride], &values[pos + stride], &valid[pos + stride],\n" 
  "      false);\n" 
  "  }\n" 
  "\n" 
  "  // Single warp per slice is completely synchronous\n" 
  "  if (Power2SortSize > 64) {\n" 
  "    barrier(CLK_LOCAL_MEM_FENCE);\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "// Sorts (key, value) pairs (in different tensors) in-place; i.e.,\n" 
  "// modifies the input `keys` and `values`\n" 
  "template <int KeyDims, int ValueDims,\n" 
  "          int Power2SortSize>\n" 
  "kernel void\n" 
  "bitonicSortKVInPlace(TensorInfo<{{IndexType}}> keys,\n" 
  "                     {{IndexType}} keySlices,\n" 
  "                     {{IndexType}} keySliceSize,\n" 
  "                     {{IndexType}} keySliceStride,\n" 
  "                     TensorInfo<{{IndexType}}> values,\n" 
  "                     {{IndexType}} valueSliceStride) {\n" 
  "  // Find the slice of the tensor that we are sorting\n" 
  "  const {{IndexType}} linearIndex = getLinearBlockId_{{IndexType}}();\n" 
  "  // Tiling the slices could have us be out of bounds, if there are a\n" 
  "  // lot of slices to sort\n" 
  "  if (linearIndex >= keySlices) {\n" 
  "    return;\n" 
  "  }\n" 
  "\n" 
  "  local {{K}} sharedKeys[{{Power2SortSize}}];\n" 
  "  local {{V}} sharedValues[{{Power2SortSize}}];\n" 
  "  local bool sharedValid[{{Power2SortSize}}];\n" 
  "\n" 
  "  const {{IndexType}} keyStartOffset =\n" 
  "    IndexToOffset_{{1000 + KeyDims}}_get(linearIndex, keys);\n" 
  "  const {{IndexType}} valueStartOffset =\n" 
  "    IndexToOffset_{{1000 + ValueDims}}_get(linearIndex, values);\n" 
  "\n" 
  "  // If the sort size is 1, the data is already sorted\n" 
  "  if (Power2SortSize == 1) {\n" 
  "    return;\n" 
  "  } else {\n" 
  "    // Otherwise, each thread is responsible for loading and storing 2\n" 
  "    // elements. The sort size is guaranteed to be >= 2\n" 
  "    const int elem1 = get_local_id(0);\n" 
  "    const int elem2 = get_local_id(0) + ({{Power2SortSize}} / 2);\n" 
  "\n" 
  "    bool valid1 = (elem1 < keySliceSize);\n" 
  "    {{K}} k1 = valid1 ?\n" 
  "      keys.data[keyStartOffset + elem1 * keySliceStride] : ({{K}}) 0;\n" 
  "    {{V}} v1 = valid1 ?\n" 
  "      values.data[valueStartOffset + elem1 * valueSliceStride] : ({{V}}) 0;\n" 
  "\n" 
  "    sharedKeys[elem1] = k1;\n" 
  "    sharedValues[elem1] = v1;\n" 
  "    sharedValid[elem1] = valid1;\n" 
  "\n" 
  "    bool valid2 = (elem2 < keySliceSize);\n" 
  "    {{K}} k2 = valid2 ?\n" 
  "      keys.data[keyStartOffset + elem2 * keySliceStride] : ({{K}}) 0;\n" 
  "    {{V}} v2 = valid2 ?\n" 
  "      values.data[valueStartOffset + elem2 * valueSliceStride] : ({{V}}) 0;\n" 
  "\n" 
  "    sharedKeys[elem2] = k2;\n" 
  "    sharedValues[elem2] = v2;\n" 
  "    sharedValid[elem2] = valid2;\n" 
  "\n" 
  "    // Sort!\n" 
  "    bitonicSort<Power2SortSize>(\n" 
  "      sharedKeys, sharedValues, sharedValid);\n" 
  "\n" 
  "    // elem1 values are always valid, since otherwise we would have\n" 
  "    // chosen the next smallest power-of-2 for sorting\n" 
  "    keys.data[keyStartOffset + elem1 * keySliceStride] =\n" 
  "      sharedKeys[elem1];\n" 
  "    values.data[valueStartOffset + elem1 * valueSliceStride] =\n" 
  "      sharedValues[elem1];\n" 
  "\n" 
  "    if (valid2) {\n" 
  "      // elem2 values might be out-of-range, if the data size we are\n" 
  "      // sorting is not a power-of-2\n" 
  "      keys.data[keyStartOffset + elem2 * keySliceStride] =\n" 
  "        sharedKeys[elem2];\n" 
  "      values.data[valueStartOffset + elem2 * valueSliceStride] =\n" 
  "        sharedValues[elem2];\n" 
  "    }\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}




#include <string>
#include <iostream>

#include "EasyCL.h"
#include "CLKernel_structs.h"
#include "util/easycl_stringhelper.h"
#include "util/StatefulTimer.h"
#include "templates/TemplatedKernel.h"

#include "THClReduceApplyUtils.h"
#include "THClSortUtils.h"
#include "THClTensorCopy.h"
#include "THClTypeParseTraits.h"
#include "THClKernels.h"

using namespace std;

static std::string getKernelTemplate();


template< typename IndexType >
void THClSortUtils_kernelLaunch_bitonicSortKVInPlace(
    THClState *state,
    dim3 grid, dim3 block,
    int KeyDims,
    int ValueDims,
    IndexType Power2SortSize,
    const TensorInfo<IndexType> &keys,
    IndexType keySlices,
    IndexType keySliceSize,
    IndexType keySliceStride,
    const TensorInfo<IndexType> &values,
    IndexType valueSliceStride,
    SortUtilsComp *comp) {
  StatefulTimer::timeCheck("bitonicSortKVInPlace START");
  std::string uniqueName = std::string("THClSortUtils_bitonicSortKVInPlace_P2S") + easycl::toString(Power2SortSize) + "_O" + comp->getOperator()
      + "_KD" +  easycl::toString(KeyDims) + "_VD" +  easycl::toString(ValueDims) + "_IT" + TypeParseTraits<IndexType>::name;

  EasyCL *cl = keys.wrapper->getCl();
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("bitonicSortKVInPlace 1aa");
  } else {
    TemplatedKernel kernelBuilder(cl);
    std::vector< int > dims;
    if( KeyDims >= 0 ) {
      dims.push_back(KeyDims);
    }
    if( ValueDims >= 0 ) {
      dims.push_back(ValueDims);
    }
    kernelBuilder
      .set("K", "float")
      .set("V", "float")
      .set("COMPARE_OP", comp->getOperator())
      .set("KeyDims", (int)KeyDims)
      .set("ValueDims", (int)ValueDims)
      .set("Power2SortSize", (int)Power2SortSize)
      .set("include_THClReduceApplyUtils", THClReduceApplyUtils_getKernelTemplate())
      .set("WarpSize", 32) // probably can do like 'if nvidia 32 else 64' ?
      .set("dims", dims)
      .set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS)
      .set("IndexType", TypeParseTraits<IndexType>::name)
    ;
    kernel = kernelBuilder.buildKernel( uniqueName, "THClSortUtils.cl", getKernelTemplate(), "bitonicSortKVInPlace" );
  }

  THClKernels k(state, kernel);
  k.inout(keys);
  k.in((int)keySlices);
  k.in((int)keySliceSize);
  k.in((int)keySliceStride);
  k.inout(values);
  k.in((int)valueSliceStride);
  k.localFloats(Power2SortSize);
  k.localFloats(Power2SortSize);
  k.localInts(Power2SortSize);

//                     local {{K}} *p_sharedKeys,
//                     local {{V}} *p_sharedValues,
//                     local bool *p_sharedValid

  k.run(grid, block);

  if(state->addFinish) cl->finish();  
  StatefulTimer::timeCheck("bitonicSortKVInPlace End");
}

std::string getKernelTemplate() {
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
  "//   dims           list of KeyDims and ValueDims\n" 
  "\n" 
  "// you need to somewhere include {{THClReduceApplyUtils}} before this, with appropriate dims, to include\n" 
  "// KeyDims and ValueDims\n" 
  "\n" 
  "{{include_THClReduceApplyUtils}}\n" 
  "\n" 
  "inline void swapVars_K(local {{K}} *p_t1, local {{K}}*p_t2) {\n" 
  "  {{K}} tmp = *p_t1;\n" 
  "  *p_t1 = *p_t2;\n" 
  "  *p_t2 = tmp;\n" 
  "}\n" 
  "\n" 
  "inline void swapVars_V(local {{V}} *p_t1, local {{V}}*p_t2) {\n" 
  "  {{V}} tmp = *p_t1;\n" 
  "  *p_t1 = *p_t2;\n" 
  "  *p_t2 = tmp;\n" 
  "}\n" 
  "\n" 
  "inline void swapVars_int(local int *p_t1, local int *p_t2) {\n" 
  "  int tmp = *p_t1;\n" 
  "  *p_t1 = *p_t2;\n" 
  "  *p_t2 = tmp;\n" 
  "}\n" 
  "\n" 
  "inline void bitonicSwap(local {{K}}* p_kA, local {{V}}*p_vA, local int*p_validA,\n" 
  "                        local {{K}}* p_kB, local {{V}}*p_vB, local int*p_validB,\n" 
  "                        int dir) {\n" 
  "  // Invalid entries always sort to the end\n" 
  "  // original cuda version was:\n" 
  "  //   int swap = (comp(kA, kB) && validA) || !validB;\n" 
  "  int swap = (((*p_kA) {{COMPARE_OP}} (*p_kB)) && (*p_validA)) || !(*p_validB);\n" 
  "  if (swap == dir) {\n" 
  "    swapVars_K(p_kA, p_kB);\n" 
  "    swapVars_V(p_vA, p_vB);\n" 
  "    swapVars_int(p_validA, p_validB);\n" 
  "  }\n" 
  "};\n" 
  "\n" 
  "inline void bitonicSort(local {{K}} *p_keys,\n" 
  "                        local {{V}} *p_values,\n" 
  "                        local int *p_valid) {\n" 
  "  #pragma unroll\n" 
  "  for (unsigned int size = 2; size < {{Power2SortSize}}; size *= 2) {\n" 
  "    int flag = ((get_local_id(0) & (size / 2)) != 0);\n" 
  "\n" 
  "    #pragma unroll\n" 
  "    for (unsigned int stride = size / 2; stride > 0; stride /= 2) {\n" 
  "\n" 
  "      // Single warp per slice is completely synchronous\n" 
  "      if ({{Power2SortSize}} > 32) {   // is 64 ok?  Let's try 32 till it is working ok...\n" 
  "        barrier(CLK_LOCAL_MEM_FENCE);\n" 
  "      }\n" 
  "\n" 
  "      unsigned int pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));\n" 
  "      bitonicSwap(\n" 
  "        p_keys + pos, p_values + pos, p_valid + pos,\n" 
  "        p_keys + pos + stride, p_values + pos + stride, p_valid + pos + stride,\n" 
  "        flag);\n" 
  "    }\n" 
  "  }\n" 
  "\n" 
  "  #pragma unroll\n" 
  "  for (unsigned int stride = {{Power2SortSize}} / 2; stride > 0; stride /= 2) {\n" 
  "    // Single warp per slice is completely synchronous\n" 
  "    if ({{Power2SortSize}} > 32) { // note: was 64 before\n" 
  "      barrier(CLK_LOCAL_MEM_FENCE);\n" 
  "    }\n" 
  "\n" 
  "    unsigned int pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));\n" 
  "    bitonicSwap(\n" 
  "      p_keys + pos, p_values + pos, p_valid + pos,\n" 
  "      p_keys + pos + stride, p_values + pos + stride, p_valid + pos + stride,\n" 
  "      false);\n" 
  "  }\n" 
  "\n" 
  "  // Single warp per slice is completely synchronous\n" 
  "  if ({{Power2SortSize}} > 32) {  // note: was 64 before\n" 
  "    barrier(CLK_LOCAL_MEM_FENCE);\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "// Sorts (key, value) pairs (in different tensors) in-place; i.e.,\n" 
  "// modifies the input `keys` and `values`\n" 
  "kernel void\n" 
  "bitonicSortKVInPlace(global TensorInfoCl *keys_info, global float *keys_data,\n" 
  "                     {{IndexType}} keySlices,\n" 
  "                     {{IndexType}} keySliceSize,\n" 
  "                     {{IndexType}} keySliceStride,\n" 
  "                     global TensorInfoCl *values_info, global float *values_data,\n" 
  "                     {{IndexType}} valueSliceStride,\n" 
  "                     local {{K}} *p_sharedKeys,\n" 
  "                     local {{V}} *p_sharedValues,\n" 
  "                     local int *p_sharedValid\n" 
  ") {\n" 
  "  // Find the slice of the tensor that we are sorting\n" 
  "  const {{IndexType}} linearIndex = getLinearBlockId();\n" 
  "  // Tiling the slices could have us be out of bounds, if there are a\n" 
  "  // lot of slices to sort\n" 
  "  if (linearIndex >= keySlices) {\n" 
  "    return;\n" 
  "  }\n" 
  "\n" 
  "//  local {{K}} sharedKeys[{{Power2SortSize}}];\n" 
  "//  local {{V}} sharedValues[{{Power2SortSize}}];\n" 
  "//  local int sharedValid[{{Power2SortSize}}];\n" 
  "\n" 
  "  const {{IndexType}} keyStartOffset =\n" 
  "    IndexToOffset_{{1000 + KeyDims}}_get(linearIndex, &keys_info[0]);\n" 
  "  const {{IndexType}} valueStartOffset =\n" 
  "    IndexToOffset_{{1000 + ValueDims}}_get(linearIndex, &values_info[0]);\n" 
  "\n" 
  "  // If the sort size is 1, the data is already sorted\n" 
  "  if ({{Power2SortSize}} == 1) {\n" 
  "    return;\n" 
  "  } else {\n" 
  "    // Otherwise, each thread is responsible for loading and storing 2\n" 
  "    // elements. The sort size is guaranteed to be >= 2\n" 
  "    const int elem1 = get_local_id(0);\n" 
  "    const int elem2 = get_local_id(0) + ({{Power2SortSize}} / 2);\n" 
  "\n" 
  "    int valid1 = (elem1 < keySliceSize);\n" 
  "    {{K}} k1 = valid1 ?\n" 
  "      keys_data[keyStartOffset + elem1 * keySliceStride] : ({{K}}) 0;\n" 
  "    {{V}} v1 = valid1 ?\n" 
  "      values_data[valueStartOffset + elem1 * valueSliceStride] : ({{V}}) 0;\n" 
  "\n" 
  "    p_sharedKeys[elem1] = k1;\n" 
  "    p_sharedValues[elem1] = v1;\n" 
  "    p_sharedValid[elem1] = valid1;\n" 
  "\n" 
  "    int valid2 = (elem2 < keySliceSize);\n" 
  "    {{K}} k2 = valid2 ?\n" 
  "      keys_data[keyStartOffset + elem2 * keySliceStride] : ({{K}}) 0;\n" 
  "    {{V}} v2 = valid2 ?\n" 
  "      values_data[valueStartOffset + elem2 * valueSliceStride] : ({{V}}) 0;\n" 
  "\n" 
  "    p_sharedKeys[elem2] = k2;\n" 
  "    p_sharedValues[elem2] = v2;\n" 
  "    p_sharedValid[elem2] = valid2;\n" 
  "\n" 
  "    // Sort!\n" 
  "//    if(get_local_id(0) == 0) {\n" 
  "    bitonicSort(p_sharedKeys, p_sharedValues, p_sharedValid);\n" 
  "//   }\n" 
  "\n" 
  "////    if(get_local_id(0) == 0) {\n" 
  "//      keys_data[0] = sharedKeys[0];\n" 
  "//      keys_data[1] = sharedKeys[1];\n" 
  "////      keys_data[0] = elem1;\n" 
  "////      keys_data[1] = elem2;\n" 
  "////      values_data[0] = {{Power2SortSize}};\n" 
  "//      values_data[0] = sharedValues[0];\n" 
  "//      values_data[1] = sharedValues[1];\n" 
  "////    }\n" 
  "\n" 
  "\n" 
  "    // elem1 values are always valid, since otherwise we would have\n" 
  "    // chosen the next smallest power-of-2 for sorting\n" 
  "    keys_data[keyStartOffset + elem1 * keySliceStride] =\n" 
  "      p_sharedKeys[elem1];\n" 
  "    values_data[valueStartOffset + elem1 * valueSliceStride] =\n" 
  "      p_sharedValues[elem1];\n" 
  "\n" 
  "    if (valid2) {\n" 
  "      // elem2 values might be out-of-range, if the data size we are\n" 
  "      // sorting is not a power-of-2\n" 
  "      keys_data[keyStartOffset + elem2 * keySliceStride] =\n" 
  "        p_sharedKeys[elem2];\n" 
  "      values_data[valueStartOffset + elem2 * valueSliceStride] =\n" 
  "        p_sharedValues[elem2];\n" 
  "    }\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}

template
void THClSortUtils_kernelLaunch_bitonicSortKVInPlace(
    THClState *state,
    dim3 grid, dim3 block,
    int KeyDims,
    int ValueDims,
    uint32 Power2SortSize,
    const TensorInfo<uint32> &keys,
    uint32 keySlices,
    uint32 keySliceSize,
    uint32 keySliceStride,
    const TensorInfo<uint32> &values,
    uint32 valueSliceStride,
    SortUtilsComp *comp);

template
void THClSortUtils_kernelLaunch_bitonicSortKVInPlace(
    THClState *state,
    dim3 grid, dim3 block,
    int KeyDims,
    int ValueDims,
    uint64 Power2SortSize,
    const TensorInfo<uint64> &keys,
    uint64 keySlices,
    uint64 keySliceSize,
    uint64 keySliceStride,
    const TensorInfo<uint64> &values,
    uint64 valueSliceStride,
    SortUtilsComp *comp);

//template< typename IndexType >
//void THClSortUtils_kernelLaunch_bitonicSortKVInPlace(
//    THClState *state,
//    dim3 grid, dim3 block,
//    int KeyDims,
//    int ValueDims,
//    int Power2SortSize,
//    const TensorInfo<IndexType> &keys,
//    IndexType keySlices,
//    IndexType keySliceSize,
//    IndexType keySliceStride,
//    const TensorInfo<IndexType> &values,
//    IndexType valueSliceStride,
//    SortUtilsComp *comp) {

//template<int>
//void kernelLaunch_bitonicSortKVInPlace(
//    THClState *state,
//    dim3 grid, dim3 block,
//    int KeyDims,
//    int ValueDims,
//    int Power2SortSize,
//    TensorInfo<IndexType> *keys,
//    IndexType keySlices,
//    IndexType keySliceSize,
//    IndexType keySliceStride,
//    TensorInfo<IndexType> *values,
//    IndexType valueSliceStride,
//    SortUtilsComp *comp);
//template EasyCL_EXPORT CLKernel *CLKernel::input(int N, const float *data);



#include "THClReduceApplyUtils.h"

#include <assert.h>
#include <stdlib.h>

// Maximum size per grid dimension that we assume
#define MAX_GRID_SIZE 65535L

void THCL_checkTensorDims(THClState* state, THClTensor* tensor, int arg) {
  long dims = THClTensor_nDimension(state, tensor);
  THArgCheck(dims <= MAX_CLTORCH_DIMS, arg, CLTORCH_DIM_WARNING);
}

bool THCL_canUse32BitIndexMath(THClState* state, THClTensor* t) {
  long elements = THClTensor_nElement(state, t);
  if (elements >= UINT_MAX) {
    return false;
  }

  long offset = 0;
  long linearId = elements - 1;

  for (int i = THClTensor_nDimension(state, t) - 1; i >= 0; --i) {
    long curDimIndex = linearId % THClTensor_size(state, t, i);
    long curDimOffset = curDimIndex * THClTensor_stride(state, t, i);
    offset += curDimOffset;
    linearId /= THClTensor_size(state, t, i);
  }

  if (offset >= UINT_MAX) {
    return false;
  }

  return true;
}

bool THCL_getGridFromTiles(long gridTiles, dim3& grid) {
  if (gridTiles > MAX_GRID_SIZE * MAX_GRID_SIZE * MAX_GRID_SIZE) {
    return false;
  }

  long gridX = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
  long gridY = 1;
  long gridZ = 1;

  if (gridTiles > MAX_GRID_SIZE) {
    gridTiles = DIVUP(gridTiles, MAX_GRID_SIZE);
    gridY = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;

    if (gridTiles > MAX_GRID_SIZE) {
      gridTiles = DIVUP(gridTiles, MAX_GRID_SIZE);
      gridZ = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
    }
  }

  grid = dim3(gridX, gridY, gridZ);
  return true;
}

namespace {
  struct SizeAndStride {
    long size;
    long stride;
  };

  int compareSizeAndStride(const void* a, const void* b) {
    const SizeAndStride* aS = (const SizeAndStride*) a;
    const SizeAndStride* bS = (const SizeAndStride*) b;

    return aS->stride < bS->stride;
  }
}

bool THCL_overlappingIndices(THClState* state, THClTensor* t) {
  // In this function, we don't care about permutations of the
  // size/stride arrays (transpositions).
  // We order the size/stride arrays by stride, skipping dimensions of
  // size 1. Strides of dimensions of size 1 don't matter, since there
  // is only one addressing point in them.
  // In this reordered view, the tensor is contiguous if
  // stride[dim] == size[dim + 1] * stride[dim + 1] for all `dim`.
  // The tensor has holes if
  // stride[dim] > size[dim + 1] * stride[dim + 1] for one or more
  // `dim`.
  // The tensor has overlaps if
  // stride[dim] < size[dim + 1] * stride[dim + 1] for one or more
  // `dim`, or the innermost stride is 0.

  // Extract size/stride arrays; only consider size >1 dims.
  SizeAndStride info[MAX_CLTORCH_DIMS];

  int dims = THClTensor_nDimension(state, t);
  int nonSize1Dims = 0;
  for (int i = 0; i < dims; ++i) {
    long size = THClTensor_size(state, t, i);
    if (size > 1) {
      info[nonSize1Dims].size = size;
      info[nonSize1Dims].stride = THClTensor_stride(state, t, i);
      ++nonSize1Dims;
    }
  }

  if (nonSize1Dims == 0) {
    // no overlap
    return false;
  }

  // Ascending order (innermost dimension in sorted view is at [0])
  qsort(info, nonSize1Dims, sizeof(SizeAndStride), compareSizeAndStride);

  // Base case: innermost dimension must have stride >= 1
  if (info[nonSize1Dims - 1].stride < 1) {
    return true;
  }

  // Subsequent dimensions, if any
  for (int i = nonSize1Dims - 2; i >= 0; --i) {
    if (info[i].stride < info[i + 1].size * info[i + 1].stride) {
      // There are overlaps
      return true;
    }
  }

  // Tensor has holes or is contiguous
  return false;
}

std::string THClReduceApplyUtils_getKernelTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClReduceApplyUtils.cl" )
  // ]]]
  // generated using cog, from THClReduceApplyUtils.cl:
  const char * kernelSource =  
  "// this needs the following template variables defined:\n" 
  "//   IndexType   string, eg 'int'\n" 
  "//   MAX_CLTORCH_DIMS    integer, eg 25\n" 
  "//   dims                list of integers, ie all dimensions >=0 this should work for\n" 
  "//   WarpSize            integer eg 32\n" 
  "//   defiscontiguous     [1|0]  (or just dont define, means 0)\n" 
  "//   defreduceblock      [1|0]  (or just dont define, means 0)\n" 
  "\n" 
  "\n" 
  "// kernel argument that defines tensor layout\n" 
  "typedef struct TensorInfoCl {\n" 
  "  // Extracts size/stride information for the kernel.\n" 
  "  // Successive dimensions can be collapsed if the size/strides match\n" 
  "  // up and thus there are no holes between the dimensions. This is used\n" 
  "  // to reduce the complexity of the problem.\n" 
  "  // The optional `reduceDim` indicates a reduction dimension for the\n" 
  "  // given tensor, so that the output size for this dimension will be 1.\n" 
  "\n" 
  "  {{IndexType}} sizes[{{MAX_CLTORCH_DIMS}}];\n" 
  "  {{IndexType}} strides[{{MAX_CLTORCH_DIMS}}];\n" 
  "  {{IndexType}} offset;\n" 
  "  int dims;\n" 
  "} TensorInfoCl;\n" 
  "// Contiguous tensors of more than one dimension are collapsed down\n" 
  "// to one tensor\n" 
  "{% if defiscontiguous==1 then %}\n" 
  "inline bool TensorInfo_isContiguous( global TensorInfoCl *tensorInfo ) {\n" 
  "    return (tensorInfo->dims == 1 && tensorInfo->strides[0] == 1);\n" 
  "}\n" 
  "{% end %}\n" 
  "\n" 
  "// Translate a linear index for the apply to a float* offset;\n" 
  "// specialized on `Dims` to reduce nvcc compilation time\n" 
  "{% for _,dim in ipairs(dims) do %}\n" 
  "inline {{IndexType}} IndexToOffset_{{1000 + dim}}_get( {{IndexType}} linearId, global TensorInfoCl *info) {\n" 
  "  {{IndexType}} offset = info->offset;\n" 
  "\n" 
  "  // Use static dims\n" 
  "//  for (int i = {{dim}} - 1; i >= 0; --i) {\n" 
  "  {{IndexType}} curDimIndex;\n" 
  "  {{IndexType}} curDimOffset;\n" 
  "  {% for i=dim-1,0,-1 do %}  // bake this in....\n" 
  "    curDimIndex = linearId % info->sizes[{{i}}];\n" 
  "    curDimOffset = curDimIndex * info->strides[{{i}}];\n" 
  "    offset += curDimOffset;\n" 
  "\n" 
  "    {% if i > 0 then %}\n" 
  "      linearId /= info->sizes[{{i}}];\n" 
  "    {% end %}\n" 
  "  {% end %}\n" 
  "//  }\n" 
  "\n" 
  "  return offset;\n" 
  "}\n" 
  "{% end %}\n" 
  "\n" 
  "inline {{IndexType}} IndexToOffset_998_get({{IndexType}} linearId, global const TensorInfoCl *info) {\n" 
  "    return linearId + info->offset;\n" 
  "}\n" 
  "\n" 
  "inline {{IndexType}} IndexToOffset_999_get({{IndexType}} linearId, global const TensorInfoCl *info) {\n" 
  "  {{IndexType}} offset = info->offset;\n" 
  "\n" 
  "  // Use dynamic dims\n" 
  "  for (int i = info->dims - 1; i >= 0; --i) {\n" 
  "    {{IndexType}} curDimIndex = linearId % info->sizes[i];\n" 
  "    {{IndexType}} curDimOffset = curDimIndex * info->strides[i];\n" 
  "    offset += curDimOffset;\n" 
  "\n" 
  "    linearId /= info->sizes[i];\n" 
  "  }\n" 
  "\n" 
  "  return offset;\n" 
  "}\n" 
  "\n" 
  "inline {{IndexType}} getLinearBlockId() {\n" 
  "  return get_group_id(2) * get_num_groups(1) * get_num_groups(0) +\n" 
  "    get_group_id(1) * get_num_groups(0) +\n" 
  "    get_group_id(0);\n" 
  "}\n" 
  "\n" 
  "// Block-wide reduction in shared memory helper; only /*threadIdx.x*/ get_local_id(0) == 0 will\n" 
  "// return the reduced value\n" 
  "{% if defreduceblock == 1 then %}\n" 
  "inline float reduceBlock( local float* smem,\n" 
  "                   int numVals,\n" 
  "                   float threadVal,\n" 
  "                   float init) {\n" 
  "  if (numVals == 0) {\n" 
  "    return init;\n" 
  "  }\n" 
  "\n" 
  "  if ((int)get_local_id(0) < numVals) {\n" 
  "    smem[ get_local_id(0)] = threadVal;\n" 
  "  }\n" 
  "\n" 
  "  // First warp will perform reductions across warps\n" 
  "  barrier(CLK_LOCAL_MEM_FENCE);\n" 
  "  if ((get_local_id(0) / {{WarpSize}}) == 0) {\n" 
  "    float r = (int)get_local_id(0) < numVals ? smem[get_local_id(0)] : init;\n" 
  "\n" 
  "    for (int i = {{WarpSize}} + get_local_id(0); i < numVals; i += {{WarpSize}}) {\n" 
  "      r = reduceOp(r, smem[i]);\n" 
  "    }\n" 
  "\n" 
  "    smem[get_local_id(0)] = r;\n" 
  "  }\n" 
  "\n" 
  "  // First thread will perform reductions across the block\n" 
  "  barrier(CLK_LOCAL_MEM_FENCE);\n" 
  "\n" 
  "  float r = init;\n" 
  "  if (get_local_id(0) == 0) {\n" 
  "    r = smem[0];\n" 
  "\n" 
  "    int numLanesParticipating = min(numVals, {{WarpSize}});\n" 
  "\n" 
  "    if (numLanesParticipating == 32) {\n" 
  "      // Unroll for {{WarpSize}} == 32 and numVals >= 32\n" 
  "      // #pragma unroll\n" 
  "      // unrolling by hand, so compiler-independent\n" 
  "      {% for i=1,31 do %}\n" 
  "        r = reduceOp(r, smem[{{i}}]);\n" 
  "      {% end %}\n" 
  "    } else {\n" 
  "      for (int i = 1; i < numLanesParticipating; ++i) {\n" 
  "        r = reduceOp(r, smem[i]);\n" 
  "      }\n" 
  "    }\n" 
  "  }\n" 
  "\n" 
  "  return r;\n" 
  "}\n" 
  "{% end %}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}




// from lib/THC/THCSortUtils.cuh:

// This needs the following template variables:
//   K              key type
//   V              value type
//   COMPARE_OP     a comparison operator, like <   or >
//   KeyDims        integer
//   ValueDims      integer
//   Power2SortSize  integer
//   dims           list of KeyDims and ValueDims

// you need to somewhere include {{THClReduceApplyUtils}} before this, with appropriate dims, to include
// KeyDims and ValueDims

{{include_THClReduceApplyUtils}}

/*__device__*/ inline void swapVars_K(local {{K}} *p_t1, local {{K}}*p_t2) {
  {{K}} tmp = *p_t1;
  *p_t1 = *p_t2;
  *p_t2 = tmp;
}

/*__device__*/ inline void swapVars_V(local {{V}} *p_t1, local {{V}}*p_t2) {
  {{V}} tmp = *p_t1;
  *p_t1 = *p_t2;
  *p_t2 = tmp;
}

/*__device__*/ inline void swapVars_bool(local bool *p_t1, local bool *p_t2) {
  bool tmp = *p_t1;
  *p_t1 = *p_t2;
  *p_t2 = tmp;
}

/*__device__*/ inline void bitonicSwap(local {{K}}* p_kA, local {{V}}*p_vA, local bool*p_validA,
                                   local {{K}}* p_kB, local {{V}}*p_vB, local bool*p_validB,
                                   bool dir) {
  // Invalid entries always sort to the end
  bool swap = ((*p_kA {{COMPARE_OP}} *p_kB) && *p_validA) || !*p_validB;
  if (swap == dir) {
    swapVars_K(p_kA, p_kB);
    swapVars_V(p_vA, p_vB);
    swapVars_bool(p_validA, p_validB);
  }
};

/*__device__*/ inline void bitonicSort(local {{K}} keys[{{Power2SortSize}}],
                                   local {{V}} values[{{Power2SortSize}}],
                                   local bool valid[{{Power2SortSize}}]) {
#pragma unroll
  for (unsigned int size = 2; size < {{Power2SortSize}}; size *= 2) {
    bool flag = ((get_local_id(0) & (size / 2)) != 0);

#pragma unroll
    for (unsigned int stride = size / 2; stride > 0; stride /= 2) {

      // Single warp per slice is completely synchronous
      if ({{Power2SortSize}} > 64) {
        barrier(CLK_LOCAL_MEM_FENCE);
      }

      unsigned int pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));
      bitonicSwap(
        &keys[pos], &values[pos], &valid[pos],
        &keys[pos + stride], &values[pos + stride], &valid[pos + stride],
        flag);
    }
  }

#pragma unroll
  for (unsigned int stride = {{Power2SortSize}} / 2; stride > 0; stride /= 2) {
    // Single warp per slice is completely synchronous
    if ({{Power2SortSize}} > 64) {
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    unsigned int pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));
    bitonicSwap(
      &keys[pos], &values[pos], &valid[pos],
      &keys[pos + stride], &values[pos + stride], &valid[pos + stride],
      false);
  }

  // Single warp per slice is completely synchronous
  if ({{Power2SortSize}} > 64) {
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

// Sorts (key, value) pairs (in different tensors) in-place; i.e.,
// modifies the input `keys` and `values`
kernel void
bitonicSortKVInPlace(global TensorInfoCl *keys_info, global float *keys_data,
                     {{IndexType}} keySlices,
                     {{IndexType}} keySliceSize,
                     {{IndexType}} keySliceStride,
                     global TensorInfoCl *values_info, global float *values_data,
                     {{IndexType}} valueSliceStride
) {
  // Find the slice of the tensor that we are sorting
  const {{IndexType}} linearIndex = getLinearBlockId_{{IndexType}}();
  // Tiling the slices could have us be out of bounds, if there are a
  // lot of slices to sort
  if (linearIndex >= keySlices) {
    return;
  }

  local {{K}} sharedKeys[{{Power2SortSize}}];
  local {{V}} sharedValues[{{Power2SortSize}}];
  local bool sharedValid[{{Power2SortSize}}];

  const {{IndexType}} keyStartOffset =
    IndexToOffset_{{1000 + KeyDims}}_get(linearIndex, &keys_info[0]);
  const {{IndexType}} valueStartOffset =
    IndexToOffset_{{1000 + ValueDims}}_get(linearIndex, &values_info[0]);

  // If the sort size is 1, the data is already sorted
  if ({{Power2SortSize}} == 1) {
    return;
  } else {
    // Otherwise, each thread is responsible for loading and storing 2
    // elements. The sort size is guaranteed to be >= 2
    const int elem1 = get_local_id(0);
    const int elem2 = get_local_id(0) + ({{Power2SortSize}} / 2);

    bool valid1 = (elem1 < keySliceSize);
    {{K}} k1 = valid1 ?
      keys_data[keyStartOffset + elem1 * keySliceStride] : ({{K}}) 0;
    {{V}} v1 = valid1 ?
      values_data[valueStartOffset + elem1 * valueSliceStride] : ({{V}}) 0;

    sharedKeys[elem1] = k1;
    sharedValues[elem1] = v1;
    sharedValid[elem1] = valid1;

    bool valid2 = (elem2 < keySliceSize);
    {{K}} k2 = valid2 ?
      keys_data[keyStartOffset + elem2 * keySliceStride] : ({{K}}) 0;
    {{V}} v2 = valid2 ?
      values_data[valueStartOffset + elem2 * valueSliceStride] : ({{V}}) 0;

    sharedKeys[elem2] = k2;
    sharedValues[elem2] = v2;
    sharedValid[elem2] = valid2;

    // Sort!
    bitonicSort(
      sharedKeys, sharedValues, sharedValid);

    // elem1 values are always valid, since otherwise we would have
    // chosen the next smallest power-of-2 for sorting
    keys_data[keyStartOffset + elem1 * keySliceStride] =
      sharedKeys[elem1];
    values_data[valueStartOffset + elem1 * valueSliceStride] =
      sharedValues[elem1];

    if (valid2) {
      // elem2 values might be out-of-range, if the data size we are
      // sorting is not a power-of-2
      keys_data[keyStartOffset + elem2 * keySliceStride] =
        sharedKeys[elem2];
      values_data[valueStartOffset + elem2 * valueSliceStride] =
        sharedValues[elem2];
    }
  }
}


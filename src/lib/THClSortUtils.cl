// from lib/THC/THCSortUtils.cuh:

/*__device__*/ inline void swapVars_K({{K}} *p_t1, {{K}}*p_t2) {
  {{K}} tmp = *p_t1;
  *p_t1 = *p_t2;
  *p_t2 = tmp;
}

/*__device__*/ inline void swapVars_V({{V}} *p_t1, {{V}}*p_t2) {
  {{V}} tmp = *p_t1;
  *p_t1 = *p_t2;
  *p_t2 = tmp;
}

/*__device__*/ inline void swapVars_bool(bool *p_t1, bool *p_t2) {
  bool tmp = *p_t1;
  *p_t1 = *p_t2;
  *p_t2 = tmp;
}

template <typename Comparator>
/*__device__*/ inline void bitonicSwap({{K}}* p_kA, V*p_vA, bool*p_validA,
                                   {{K}}* p_kB, V*p_vB, bool*p_validB,
                                   bool dir,
                                   const Comparator& comp) {
  // Invalid entries always sort to the end
  bool swap = (comp(kA, kB) && validA) || !validB;
  if (swap == dir) {
    swapVars_K(p_kA, p_kB);
    swapVars_V(p_vA, p_vB);
    swapVars_bool(p_validA, p_validB);
  }
};

template <typename Comparator, 
          typename {{IndexType}}, int Power2SortSize>
/*__device__*/ inline void bitonicSort({{K}} keys[Power2SortSize],
                                   {{V}} values[Power2SortSize],
                                   bool valid[Power2SortSize],
                                   const Comparator& comp) {
#pragma unroll
  for (unsigned int size = 2; size < Power2SortSize; size *= 2) {
    bool flag = ((get_local_id(0) & (size / 2)) != 0);

#pragma unroll
    for (unsigned int stride = size / 2; stride > 0; stride /= 2) {

      // Single warp per slice is completely synchronous
      if (Power2SortSize > 64) {
        barrier(CLK_LOCAL_MEM_FENCE);
      }

      unsigned int pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));
      bitonicSwap<Comparator>(
        &keys[pos], &values[pos], &valid[pos],
        &keys[pos + stride], &values[pos + stride], &valid[pos + stride],
        flag, comp);
    }
  }

#pragma unroll
  for (unsigned int stride = Power2SortSize / 2; stride > 0; stride /= 2) {
    // Single warp per slice is completely synchronous
    if (Power2SortSize > 64) {
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    unsigned int pos = 2 * get_local_id(0) - (get_local_id(0) & (stride - 1));
    bitonicSwap<Comparator>(
      &keys[pos], &values[pos], &valid[pos],
      &keys[pos + stride], &values[pos + stride], &valid[pos + stride],
      false, comp);
  }

  // Single warp per slice is completely synchronous
  if (Power2SortSize > 64) {
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

// Sorts (key, value) pairs (in different tensors) in-place; i.e.,
// modifies the input `keys` and `values`
template <int KeyDims, int ValueDims,
          typename Comparator, typename {{IndexType}}, int Power2SortSize>
kernel void
bitonicSortKVInPlace(TensorInfo<{{IndexType}}> keys,
                     {{IndexType}} keySlices,
                     {{IndexType}} keySliceSize,
                     {{IndexType}} keySliceStride,
                     TensorInfo<{{IndexType}}> values,
                     {{IndexType}} valueSliceStride,
                     const Comparator& comp) {
  // Find the slice of the tensor that we are sorting
  const {{IndexType}} linearIndex = getLinearBlockId<{{IndexType}}>();
  // Tiling the slices could have us be out of bounds, if there are a
  // lot of slices to sort
  if (linearIndex >= keySlices) {
    return;
  }

  local {{K}} sharedKeys[Power2SortSize];
  local {{V}} sharedValues[Power2SortSize];
  local bool sharedValid[Power2SortSize];

  const {{IndexType}} keyStartOffset =
    IndexToOffset<{{IndexType}}, KeyDims>::get(linearIndex, keys);
  const {{IndexType}} valueStartOffset =
    IndexToOffset<{{IndexType}}, ValueDims>::get(linearIndex, values);

  // If the sort size is 1, the data is already sorted
  if (Power2SortSize == 1) {
    return;
  } else {
    // Otherwise, each thread is responsible for loading and storing 2
    // elements. The sort size is guaranteed to be >= 2
    const int elem1 = get_local_id(0);
    const int elem2 = get_local_id(0) + (Power2SortSize / 2);

    bool valid1 = (elem1 < keySliceSize);
    {{K}} k1 = valid1 ?
      keys.data[keyStartOffset + elem1 * keySliceStride] : ({{K}}) 0;
    {{V}} v1 = valid1 ?
      values.data[valueStartOffset + elem1 * valueSliceStride] : ({{V}}) 0;

    sharedKeys[elem1] = k1;
    sharedValues[elem1] = v1;
    sharedValid[elem1] = valid1;

    bool valid2 = (elem2 < keySliceSize);
    {{K}} k2 = valid2 ?
      keys.data[keyStartOffset + elem2 * keySliceStride] : ({{K}}) 0;
    {{V}} v2 = valid2 ?
      values.data[valueStartOffset + elem2 * valueSliceStride] : ({{V}}) 0;

    sharedKeys[elem2] = k2;
    sharedValues[elem2] = v2;
    sharedValid[elem2] = valid2;

    // Sort!
    bitonicSort<Comparator, K, V, {{IndexType}}, Power2SortSize>(
      sharedKeys, sharedValues, sharedValid, comp);

    // elem1 values are always valid, since otherwise we would have
    // chosen the next smallest power-of-2 for sorting
    keys.data[keyStartOffset + elem1 * keySliceStride] =
      sharedKeys[elem1];
    values.data[valueStartOffset + elem1 * valueSliceStride] =
      sharedValues[elem1];

    if (valid2) {
      // elem2 values might be out-of-range, if the data size we are
      // sorting is not a power-of-2
      keys.data[keyStartOffset + elem2 * keySliceStride] =
        sharedKeys[elem2];
      values.data[valueStartOffset + elem2 * valueSliceStride] =
        sharedValues[elem2];
    }
  }
}


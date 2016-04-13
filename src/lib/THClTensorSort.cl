// from lib/THC/THCTensorSort.cu:

// needs following tmeplate variables defined:
//  Dim      integer
//  IndexType  string 'int'

{{include_THClReduceApplyUtils}}

{{include_THClSortUtils}}

// `base` is the base address of a tensor
// For each slice (defined as a linear point of `out`, from 0 ->
// (sliceSize - 1) * sliceStride, we fill that slice from `0` to
// `sliceSize - 1`.
kernel void
fillSliceWithIndex(global TensorInfoCl *out_info, global float *out_data,
                   {{IndexType}} totalSlices,
                   {{IndexType}} sliceSize,
                   {{IndexType}} sliceStride) {
  {{IndexType}} slice = getLinearBlockId();

  if (slice >= totalSlices) {
    return;
  }

  const unsigned long offset =
    IndexToOffset_{{1000+Dim}}_get(slice, &out_info[0]);
  float* base = &out_data[offset];

  for (long i = get_local_id(0); i < sliceSize; i += get_local_size(0)) {
    // Torch indices are 1-based (hence the +1)
    base[i * sliceStride] = (float) i + 1.0f;
  }
}


// from lib/THC/THCTensorSort.cu:

{{include_THClReduceApplyUtils}}

// `base` is the base address of a tensor
// For each slice (defined as a linear point of `out`, from 0 ->
// (sliceSize - 1) * sliceStride, we fill that slice from `0` to
// `sliceSize - 1`.
template <typename {{IndexType}}, int Dim>
kernel void
fillSliceWithIndex(TensorInfo<{{IndexType}}> out,
                   {{IndexType}} totalSlices,
                   {{IndexType}} sliceSize,
                   {{IndexType}} sliceStride) {
  {{IndexType}} slice = getLinearBlockId<{{IndexType}}>();

  if (slice >= totalSlices) {
    return;
  }

  const unsigned long offset =
    IndexToOffset<{{IndexType}}, Dim>::get(slice, out);
  float* base = &out.data[offset];

  for (long i = get_local_id(0); i < sliceSize; i += get_local_size(0)) {
    // Torch indices are 1-based (hence the +1)
    base[i * sliceStride] = (float) i + 1.0f;
  }
}


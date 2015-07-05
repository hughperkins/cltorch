// Threads per thread block
#define THCL_NONCONTIG_REDUCE_BLOCK_SIZE 32 * 16

static float modifyOp(float _in1) {
  float _out;
  float *in1 = &_in1;
  float *out = &_out;
  {{modify_operation}};
  return _out;
}

static float reduceOp(float _in1, float _in2) {
  // I guess the compiler can sort this stuff out :-P
  float _out;
  float *in1 = &_in1;
  float *in2 = &_in2;
  float *out = &_out;
  {{reduce_operation}};
  return _out;
}

{{include_THClReduceApplyUtils}}

static {{IndexType}} getReduceNoncontigDimSliceIndex() {
  // Each thread handles one slice
  return getLinearBlockId() * THCL_NONCONTIG_REDUCE_BLOCK_SIZE + /*threadIdx.x*/ get_local_id(0);
}

// Kernel that handles an entire reduction of a slice of a tensor per each thread
kernel void
THClTensor_reduceNoncontigDim(global TensorInfoCl *out_info,
                              global float *out_data,
                              global TensorInfoCl *in_info,
                              global float *in_data,
                              int reductionStride,
                              int reductionSize,
                              int totalSlices,
                              float init) {
  const {{IndexType}} sliceIndex = getReduceNoncontigDimSliceIndex();

  if (sliceIndex >= totalSlices) {
    return;
  }

  // Each thread picks a point in `out` and `in` for which it is
  // producing the reduction
  const {{IndexType}} outOffset =
    IndexToOffset_{{1000 + dim1}}_get(sliceIndex, out_info[0]);
  const {{IndexType}} inBaseOffset =
    IndexToOffset_{{1000 + dim2}}_get(sliceIndex, in_info[0]);

  // For each point in reductionSize, reduce into `r`
  {{IndexType}} inOffset = inBaseOffset;
  float r = init;

  for ({{IndexType}} i = 0; i < reductionSize; ++i) {
    r = reduceOp(r, modifyOp(in_data[inOffset]));
    inOffset += reductionStride;
  }

  // Write out reduced value
  out_data[outOffset] = r;
}

static {{IndexType}} getReduceContigDimSliceIndex() {
  // Each block handles one slice
  return getLinearBlockId();
}

// Kernel that handles an entire reduction of a slice of a tensor per
// each block
kernel void
THClTensor_reduceContigDim(global TensorInfoCl *out_info,
                           global float *out_data,
                           global TensorInfoCl *in_info,
                           global float *in_data,
                           int reductionSize,
                           int totalSlices,
                           float init,
                           local float *smem) {
  const {{IndexType}} sliceIndex = getReduceContigDimSliceIndex();

  if (sliceIndex >= totalSlices) {
    return;
  }

  // Get the offset in `out` for the reduction
  const {{IndexType}} outOffset =
    IndexToOffset_{{1000 + dim1}}_get(sliceIndex, out_info[0]);

  // Get the base offset in `in` for this block's reduction
  const {{IndexType}} inBaseOffset =
    IndexToOffset_{{1000 + dim2}}_get(sliceIndex, in_info[0]);

  // Each thread in the block will reduce some subset of elements in
  // the slice. The elements are guaranteed contiguous starting at
  // `inBaseOffset`.
  float r = init;
  for ({{IndexType}} i = /*threadIdx.x*/ get_local_id(0); i < reductionSize; i += /*blockDim.x*/ get_local_size(0)) {
    r = reduceOp(r, modifyOp(in_data[inBaseOffset + i]));
  }

  // Reduce within the block
//  extern __shared__ float smem[];
  r = reduceBlock(smem, /*blockDim.x*/ get_local_size(0), r, init);

  if (/*threadIdx.x*/ get_local_id(0) == 0) {
    // Write out reduced value
    out_data[outOffset] = r;
  }
}


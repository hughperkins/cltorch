{{include_THClDeviceUtils}}

static inline float modifyOp(float _in1) {
  float _out;
  float *in1 = &_in1;
  float *out = &_out;
  {{modify_operation}};
  return _out;
}

static inline float reduceOp(float _in1, float _in2) {
  // I guess the compiler can sort this stuff out :-P
  float _out;
  float *in1 = &_in1;
  float *in2 = &_in2;
  float *out = &_out;
  {{reduce_operation}};
  return _out;
}

{{include_THClReduceApplyUtils}}

// Kernel that handles an entire reduction of a tensor in one pass
kernel void
THClTensor_reduceAll(global TensorInfoCl *in_info,
                     global float *in_data,
                     {{IndexType}} totalElements,
                     float init,
                     global float* out,
                     local float *smem) {
  // With a block-wide stride, have each thread perform its own reduction.
  float r = init;
  for ({{IndexType}} i = get_local_id(0); i < totalElements; i += get_local_size(0)) {
    const {{IndexType}} inOffset = IndexToOffset_{{1000 + dim1}}_get(i, &in_info[0]);
    r = reduceOp(r, modifyOp(in_data[inOffset]));
  }

  // Reduce within the block
  r = reduceBlock(smem, get_local_size(0), r, init);

  if(get_local_id(0) == 0) {
    // Write out reduced value
    out[0] = r;
  }
}

static inline {{IndexType}} getStartIndex({{IndexType}} totalSize) {
  {{IndexType}} sizePerBlock = THClCeilDiv(totalSize, ({{IndexType}}) get_num_groups(0));
  return get_group_id(0) * sizePerBlock;
}

static inline {{IndexType}} getEndIndex({{IndexType}} totalSize) {
  {{IndexType}} sizePerBlock = THClCeilDiv(totalSize, ({{IndexType}}) get_num_groups(0));
  return min(({{IndexType}}) ((get_group_id(0) + 1) * sizePerBlock), totalSize);
}

// Kernel that handles an entire reduction of a tensor in two passes
kernel void
THClTensor_reduceAllPass1(global TensorInfoCl *in_info,
                          global float *in_data,
                          {{IndexType}} totalElements,
                          float init,
                          global float* scratchSpace,
                          local float *smem) {
  const {{IndexType}} startIndex = getStartIndex(totalElements);
  const {{IndexType}} endIndex = getEndIndex(totalElements);

  // With a block-wide stride, have each thread perform its own reduction.
  float r = init;
  for ({{IndexType}} i = startIndex + get_local_id(0); i < endIndex; i += get_local_size(0)) {
    const {{IndexType}} inOffset = IndexToOffset_{{1000 + dim1}}_get(i, &in_info[0]);
    r = reduceOp(r, modifyOp(in_data[inOffset]));
  }

  // Reduce within the block
  r = reduceBlock(smem, get_local_size(0), r, init);

  if ((int)get_local_id(0) == 0) {
    // Write out block-wide reduced value
    scratchSpace[get_group_id(0)] = r;
  }
}

kernel void THClTensor_reduceAllPass2(int numPass1Blocks,
                            float init,
                            global float* scratchSpace,
                            global float* out,
                            local float *smem) {
  float r = init;
  if ((int)get_local_id(0) < numPass1Blocks) {
    r = scratchSpace[get_local_id(0)];
  }

  // Reduce within the block
  r = reduceBlock(smem, numPass1Blocks, r, init);

  if((int)get_local_id(0) == 0) {
    out[0] = r;
  }
}




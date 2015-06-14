{{include_THClDeviceUtils}}

float modifyOp(float _in1) {
  float _out;
  float *in1 = &_in1;
  float *out = &_out;
  {{modify_operation}};
  return _out;
}

float reduceOp(float _in1, float _in2) {
  // I guess the compiler can sort this stuff out :-P
  float _out;
  float *in1 = &_in1;
  float *in2 = &_in2;
  float *out = &_out;
  {{reduce_operation}};
  return _out;
}

{{IndexType}} getEndIndex({{IndexType}} totalSize) {
  {{IndexType}} sizePerBlock = THClCeilDiv(totalSize, ({{IndexType}}) gridDim.x);
  return min(({{IndexType}}) ((blockIdx.x + 1) * sizePerBlock), totalSize);
}

// Kernel that handles an entire reduction of a tensor in one pass
kernel void
THClTensor_reduceAll(global TensorInfoCl *in,
                     global float *in_data,
                     {{IndexType}} totalElements,
                     float init,
                     global float* out,
                     local float *smem) {
  // With a block-wide stride, have each thread perform its own reduction.
  float r = init;
  for ({{IndexType}} i = threadIdx.x; i < totalElements; i += blockDim.x) {
    const {{IndexType}} inOffset = IndexToOffset_{{1000 + dim1}}_get(i, in);
    r = reduceOp(r, modifyOp(in.data[inOffset]));
  }

  // Reduce within the block
//  extern __shared__ float smem[];
  r = reduceBlock(smem, blockDim.x, r, reduceOp, init);

  if (threadIdx.x == 0) {
    // Write out reduced value
    out[0] = r;
  }
}

// Kernel that handles an entire reduction of a tensor in two passes
kernel void
THClTensor_reduceAllPass1(global TensorInfoCl *in,
                          global float *in_data,
                          {{IndexType}} totalElements,
                          float init,
                          global float* scratchSpace,
                          local float *smem) {
  const {{IndexType}} startIndex = getStartIndex<{{IndexType}}>(totalElements);
  const {{IndexType}} endIndex = getEndIndex<{{IndexType}}>(totalElements);

  // With a block-wide stride, have each thread perform its own reduction.
  float r = init;
  for ({{IndexType}} i = startIndex + threadIdx.x; i < endIndex; i += blockDim.x) {
    const {{IndexType}} inOffset = IndexToOffset_{{1000 + dim1}}_get(i, in);
    r = reduceOp(r, modifyOp(in.data[inOffset]));
  }

  // Reduce within the block
//  extern __shared__ float smem[];
  r = reduceBlock(smem, blockDim.x, r, init);

  if (threadIdx.x == 0) {
    // Write out block-wide reduced value
    scratchSpace[blockIdx.x] = r;
  }
}

template <typename ReduceOp, typename {{IndexType}}>
kernel THClTensor_reduceAllPass2(int numPass1Blocks,
                            float init,
                            global float* scratchSpace,
                            global float* out,
                            local float *smem) {
  float r = init;
  if (threadIdx.x < numPass1Blocks) {
    r = scratchSpace[threadIdx.x];
  }

  // Reduce within the block
//  extern __shared__ float smem[];
  r = reduceBlock(smem, numPass1Blocks, r, init);

  if (threadIdx.x == 0) {
    out[0] = r;
  }
}




IndexType getStartIndex(IndexType totalSize) {
  IndexType sizePerBlock = THCCeilDiv(totalSize, (IndexType) gridDim.x);
  return blockIdx.x * sizePerBlock;
}

IndexType getEndIndex(IndexType totalSize) {
  IndexType sizePerBlock = THCCeilDiv(totalSize, (IndexType) gridDim.x);
  return min((IndexType) ((blockIdx.x + 1) * sizePerBlock), totalSize);
}

// Kernel that handles an entire reduction of a tensor in one pass
kernel void
THClTensor_reduceAll(global TensorInfoCl *in,
                     global float *in_data,
                     IndexType totalElements,
                     float init,
                     global float* out,
                     local float *smem) {
  // With a block-wide stride, have each thread perform its own reduction.
  float r = init;
  for (IndexType i = threadIdx.x; i < totalElements; i += blockDim.x) {
    const IndexType inOffset = IndexToOffset<IndexType, ADims>::get(i, in);
    r = reduceOp(r, modifyOp(in.data[inOffset]));
  }

  // Reduce within the block
//  extern __shared__ float smem[];
  r = reduceBlock<float, ReduceOp>(smem, blockDim.x, r, reduceOp, init);

  if (threadIdx.x == 0) {
    // Write out reduced value
    out[0] = r;
  }
}

// Kernel that handles an entire reduction of a tensor in two passes
kernel void
THClTensor_reduceAllPass1(global TensorInfoCl *in,
                          global float *in_data,
                          IndexType totalElements,
                          float init,
                          global float* scratchSpace,
                          local float *smem) {
  const IndexType startIndex = getStartIndex<IndexType>(totalElements);
  const IndexType endIndex = getEndIndex<IndexType>(totalElements);

  // With a block-wide stride, have each thread perform its own reduction.
  float r = init;
  for (IndexType i = startIndex + threadIdx.x; i < endIndex; i += blockDim.x) {
    const IndexType inOffset = IndexToOffset<IndexType, ADims>::get(i, in);
    r = reduceOp(r, modifyOp(in.data[inOffset]));
  }

  // Reduce within the block
//  extern __shared__ float smem[];
  r = reduceBlock<float, ReduceOp>(smem, blockDim.x, r, reduceOp, init);

  if (threadIdx.x == 0) {
    // Write out block-wide reduced value
    scratchSpace[blockIdx.x] = r;
  }
}

template <typename ReduceOp, typename IndexType>
__global__ void
THClTensor_reduceAllPass2(int numPass1Blocks,
                            float init,
                            ReduceOp reduceOp,
                            float* scratchSpace,
                            float* out) {
  float r = init;
  if (threadIdx.x < numPass1Blocks) {
    r = scratchSpace[threadIdx.x];
  }

  // Reduce within the block
  extern __shared__ float smem[];
  r = reduceBlock<float, ReduceOp>(smem, numPass1Blocks, r, reduceOp, init);

  if (threadIdx.x == 0) {
    *out = r;
  }
}




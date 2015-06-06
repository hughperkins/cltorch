# cutorch-apply

## THCApply

```
typedef struct THCudaTensor
{
    long *size;
    long *stride;
    int nDimension;

    THCudaStorage *storage;
    long storageOffset;
    int refcount;

    char flag;

} THCudaTensor;
```

From THCReduceApplyUtils.h:
```
#define MAX_CUTORCH_DIMS 25
```

From THCReduceApplyUtils.h:
```
// CUDA kernel argument that defines tensor layout
template <typename IndexType>
struct TensorInfo {
  // Extracts size/stride information for the kernel.
  // Successive dimensions can be collapsed if the size/strides match
  // up and thus there are no holes between the dimensions. This is used
  // to reduce the complexity of the problem.
  // The optional `reduceDim` indicates a reduction dimension for the
  // given tensor, so that the output size for this dimension will be 1.
  TensorInfo(THCState* state, THCudaTensor* t, int reduceDim = -1);

  // Contiguous tensors of more than one dimension are collapsed down
  // to one tensor
  __host__ __device__ inline bool isContiguous() const {
    return (dims == 1 && strides[0] == 1);
  }

  float* data;
  IndexType sizes[MAX_CUTORCH_DIMS];
  IndexType strides[MAX_CUTORCH_DIMS];
  int dims;
};
```

```
template <typename Op>
bool THCudaTensor_pointwiseApply3(THCState* state,
                                  THCudaTensor* a,
                                  THCudaTensor* b,
                                  THCudaTensor* c,
                                  const Op& op,
                                  TensorArgType aType = ReadWrite,
                                  TensorArgType bType = ReadOnly,
                                  TensorArgType cType = ReadOnly)
```

```
  THCudaTensor_pointwiseApply3<Op, TYPE, A, B, C>
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      aInfo, bInfo, cInfo, (TYPE) totalElements, op);
```

```
template <typename Op, typename IndexType, int ADims, int BDims, int CDims>
__global__ void
THCudaTensor_pointwiseApply3(TensorInfo<IndexType> a,
                             TensorInfo<IndexType> b,
                             TensorInfo<IndexType> c,
                             IndexType totalElements,
                             Op op)
```


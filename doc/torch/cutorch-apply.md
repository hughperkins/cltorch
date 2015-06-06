# cutorch-apply

## THCApply

Useful CUDA intro :-P http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf
- `__global__` is a kernel, equivalent to OpenCL `kernel`
- `mykernel<<<num_workgroups, workgroup_size>>>(param1, param2, ...)` , with the triple brackets, is a *kernel launch* (equivalent to OpenCL `run(dims, num_workgroups * workgroup_size, workgroup_size)` (ish...)
- `__shared__` means local memory, ie `__local__` in OpenCL
- `__syncthreads()` is like `barrier(CLK_LOCAL_MEM_FENCE)` in OpenCL
- `cudaDeviceSynchronize()` is like `clFinish()`
- `__device__` means a function that can be called from a kernel
- `__host__` means a function that can be called from the host, ie from c/c++ main program
  - possible to add both `__device__` and `__host__`, just to be really confusing :-P

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
  // note: since both __host__ and __device__, this is available from both main
  // c++ code, and from kernels
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
// Translate a linear index for the apply to a float* offset;
// specialized on `Dims` to reduce nvcc compilation time
template <typename IndexType, int Dims>
struct IndexToOffset {
  static __host__ __device__ IndexType get(
    IndexType linearId,
    const TensorInfo<IndexType>& info) {
    IndexType offset = 0;

    // Use static dims
    for (int i = Dims - 1; i >= 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;

      if (i > 0) {
        linearId /= info.sizes[i];
      }
    }

    return offset;
  }
};
```

```
// This is the kernel entry point, since it is marked with `__global__`
template <typename Op, typename IndexType, int ADims, int BDims, int CDims>
__global__ void
THCudaTensor_pointwiseApply3(TensorInfo<IndexType> a,
                             TensorInfo<IndexType> b,
                             TensorInfo<IndexType> c,
                             IndexType totalElements,
                             Op op)
```

```
// This is a normal C++ host-side method, not kernel or anything
// It happens to launch the kernel though, ie launches 
// THCudaTensor_pointwiseApply3, above
template <typename Op>
bool THCudaTensor_pointwiseApply3(THCState* state,
                                  THCudaTensor* a,
                                  THCudaTensor* b,
                                  THCudaTensor* c,
                                  const Op& op,
                                  TensorArgType aType = ReadWrite,
                                  TensorArgType bType = ReadOnly,
                                  TensorArgType cType = ReadOnly) {
  ...
  // triple quotes, so this is a kernel *launch*
  THCudaTensor_pointwiseApply3<Op, TYPE, A, B, C>
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
      aInfo, bInfo, cInfo, (TYPE) totalElements, op);
  ...
}
```


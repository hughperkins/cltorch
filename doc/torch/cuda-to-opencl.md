# CUDA to OpenCL

Useful CUDA intro/info:
- http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf
- http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels
- http://developer.amd.com/tools-and-sdks/opencl-zone/opencl-resources/programming-in-opencl/porting-cuda-applications-to-opencl/

notes:
- `__global__` is a kernel, equivalent to OpenCL `kernel`
- `mykernel<<<num_workgroups, workgroup_size>>>(param1, param2, ...)` , with the triple brackets, is a *kernel launch* (equivalent to OpenCL `run(dims, num_workgroups * workgroup_size, workgroup_size)` (ish...)
  - `num_workgroups` and `workgroup_size` can be integers, or `dim3`
  - where there are 4 launch parameters, the fourth is the stream, ie `<<<num_workgroups, workgroup_size, 0, stream>>>`
- `__shared__` means local memory, ie `__local__` in OpenCL
- `__syncthreads()` is like `barrier(CLK_LOCAL_MEM_FENCE)` in OpenCL
- `cudaDeviceSynchronize()` is like `clFinish()`
- `__device__` means a function that can be called from a kernel
- `__host__` means a function that can be called from the host, ie from c/c++ main program
  - possible to add both `__device__` and `__host__`, just to be really confusing :-P

## Indexing

|CUDA | OpenCL |
|---|---|
|gridDim | get_num_groups() |
| blockDim | get_local_size() |
| blockIdx | get_group_id() |
| threadIdx | get_local_id() |
|   | get_global_id() |
|   | get_global_size() |



// from lib/THC/THCTensorIndex.cu:

kernel void THClTensor_kernel_indexFill(
   global float *tensor_data, int tensor_offset,
  global int* stride,
  global float *index_data, int index_offset, 
  int src_nDim,
   int dim, int idx_size, int tensor_size, int size_dim, float val
)
{
  int thread_idx = get_group_id(0) * get_local_size(0) * get_local_size(1) + get_local_id(1) * get_local_size(0) + get_local_id(0);

  long flat_size = tensor_size / idx_size;

  if (thread_idx < flat_size)
  {
    long coeff = 0;
    for (int i=0; i<idx_size; i++)
    {
      int leftover = thread_idx;
      int srcIdx = 0;
      for (int d=0; d<src_nDim; d++)
      {
        if (d < dim)
        {
          coeff = leftover / (stride[d] / size_dim);
          leftover -= coeff * (stride[d] / size_dim);
          srcIdx += coeff * stride[d];
        }
        else if (d > dim)
        {
          coeff = leftover / stride[d];
          leftover -= coeff * stride[d];
          srcIdx += coeff * stride[d];
        }
      }
      tensor_data[tensor_offset + srcIdx + (int)((index_data[index_offset + i])-1)*stride[dim] ] = val;
    }
  }
}

kernel void THClTensor_kernel_indexCopy(
   global float *res_data, int res_offset, 
   global float *src_data, int src_offset,
   global int* res_stride, global float *index_data, int index_offset,
   int res_nDim, int dim, int idx_size, int src_size, int size_dim
)
{
  int thread_idx = get_group_id(0) * get_local_size(0) * get_local_size(1) + get_local_id(1) * get_local_size(0) + get_local_id(0);

  long flat_size = src_size / idx_size;

  if (thread_idx < flat_size)
  {
    long coeff = 0;
    for (int i=0; i<idx_size; i++)
    {
      int leftover = thread_idx;
      int targetIdx = 0;
      int resIdx = 0;
      for (int d=0; d<res_nDim; d++)
      {
        if (d < dim)
        {
          long stride_d = res_stride[d] / size_dim;
          coeff = leftover / stride_d;
          leftover -= coeff * stride_d;
          targetIdx += coeff * stride_d * idx_size;
          resIdx += coeff * res_stride[d];
        }
        else if (d > dim)
        {
          coeff = leftover / res_stride[d];
          leftover -= coeff * res_stride[d];
          targetIdx += coeff * res_stride[d];
          resIdx += coeff * res_stride[d];
        }
      }
      res_data[res_offset + resIdx + ((int)(index_data[index_offset + i])-1)*res_stride[dim] ] = src_data[src_offset + targetIdx + i*res_stride[dim] ];
    }
  }
}

kernel void THClTensor_kernel_indexSelect(
   global float *tensor_data, long tensor_offset, global float *src_data, long src_offset,
  global long* src_stride, global float *index_data, long index_offset,
   long src_nDim, int dim, long idx_size, long tensor_size, long size_dim
)
{
  int thread_idx = get_group_id(0) * get_local_size(0) * get_local_size(1) + get_local_id(1) * get_local_size(0) + get_local_id(0);

  long flat_size = tensor_size / idx_size;

  if (thread_idx < flat_size)
  {
    long coeff = 0;
    for (int i=0; i<idx_size; i++)
    {
      int leftover = thread_idx;
      int targetIdx = 0;
      int srcIdx = 0;
      for (int d=0; d<src_nDim; d++)
      {
        if (d < dim)
        {
          long stride_d = src_stride[d] / size_dim;
          coeff = leftover / stride_d;
          leftover -= coeff * stride_d;
          targetIdx += coeff * stride_d * idx_size;
          srcIdx += coeff * src_stride[d];
        }
        else if (d > dim)
        {
          coeff = leftover / src_stride[d];
          leftover -= coeff * src_stride[d];
          targetIdx += coeff * src_stride[d];
          srcIdx += coeff * src_stride[d];
        }
      }
      tensor_data[tensor_offset + targetIdx + i*src_stride[dim] ] = src_data[src_offset + srcIdx + ((int)(index_data[index_offset + i])-1)*src_stride[dim] ];
    }
  }
}


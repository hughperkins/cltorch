typedef struct TensorInfoCl2 {
  {{IndexType}} stride[{{num_dims}}];
  {{IndexType}} size[{{num_dims}}];
  {{IndexType}} offset;
} TensorInfoCl2;

kernel void THClTensor_kernel_gather(
    global TensorInfoCl *tgt_info, global float*tgt_data,
    global TensorInfoCl *src_info, global float*src_data,
   int dim,
    global TensorInfoCl *idx_info, global float*idx_data,
   int totalElements
)
{
  for (int linearIndex = get_global_id(0);
       linearIndex < totalElements;
       linearIndex += get_global_size(0)) {
     const int idx_offset =
      IndexToOffset_{{1000 + dim}}_get(linearIndex, tgt_info[0]);
//     long idx = idx_data[

     const int tgt_offset =
      IndexToOffset_{{1000 + dim}}_get(linearIndex, tgt_info[0]);
     const int src_offset =
      IndexToOffset_{{1000 + dim}}_get(linearIndex, tgt_info[0]);
  }
}


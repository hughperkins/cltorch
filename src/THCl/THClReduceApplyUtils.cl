// kernel argument that defines tensor layout
typedef struct TensorInfoCl {
  // Extracts size/stride information for the kernel.
  // Successive dimensions can be collapsed if the size/strides match
  // up and thus there are no holes between the dimensions. This is used
  // to reduce the complexity of the problem.
  // The optional `reduceDim` indicates a reduction dimension for the
  // given tensor, so that the output size for this dimension will be 1.

  {{IndexType}} sizes[{{MAX_CLTORCH_DIMS}}];
  {{IndexType}} strides[{{MAX_CLTORCH_DIMS}}];
  {{IndexType}} offset;
  int dims;
} TensorInfoCl;
// Contiguous tensors of more than one dimension are collapsed down
// to one tensor
inline bool TensorInfo_isContiguous( TensorInfoCl tensorInfo ) {
    return (tensorInfo.dims == 1 && tensorInfo.strides[0] == 1);    
}

// Translate a linear index for the apply to a float* offset;
// specialized on `Dims` to reduce nvcc compilation time
{% for _,dim in ipairs(dims) do %}
inline {{IndexType}} IndexToOffset_{{1000 + dim}}_get( {{IndexType}} linearId, TensorInfoCl info) {
  {{IndexType}} offset = info.offset;

  // Use static dims
//  for (int i = {{dim}} - 1; i >= 0; --i) {
  {{IndexType}} curDimIndex;
  {{IndexType}} curDimOffset;
  {% for i=dim-1,0,-1 do %}  // bake this in....
    curDimIndex = linearId % info.sizes[{{i}}];
    curDimOffset = curDimIndex * info.strides[{{i}}];
    offset += curDimOffset;

    {% if i > 0 then %}
      linearId /= info.sizes[{{i}}];
    {% end %}
  {% end %}
//  }

  return offset;
}
{% end %}

inline {{IndexType}} IndexToOffset_998_get({{IndexType}} linearId, const TensorInfoCl info) {
    return linearId + info.offset;
}

inline {{IndexType}} IndexToOffset_999_get({{IndexType}} linearId, const TensorInfoCl info) {
  {{IndexType}} offset = info.offset;

  // Use dynamic dims
  for (int i = info.dims - 1; i >= 0; --i) {
    {{IndexType}} curDimIndex = linearId % info.sizes[i];
    {{IndexType}} curDimOffset = curDimIndex * info.strides[i];
    offset += curDimOffset;

    linearId /= info.sizes[i];
  }

  return offset;
}

inline {{IndexType}} getLinearBlockId() {
  return get_group_id(2) * get_num_groups(1) * get_num_groups(0) +
    get_group_id(1) * get_num_groups(0) +
    get_group_id(0);
}

// Block-wide reduction in shared memory helper; only /*threadIdx.x*/ get_local_id(0) == 0 will
// return the reduced value
inline float reduceBlock( local float* smem,
                   int numVals,
                   float threadVal,
                   float init) {
  if (numVals == 0) {
    return init;
  }

  if ((int)get_local_id(0) < numVals) {
    smem[ get_local_id(0)] = threadVal;
  }

  // First warp will perform reductions across warps
  barrier(CLK_LOCAL_MEM_FENCE);
  if ((get_local_id(0) / {{WarpSize}}) == 0) {
    float r = (int)get_local_id(0) < numVals ? smem[get_local_id(0)] : init;

    for (int i = {{WarpSize}} + get_local_id(0); i < numVals; i += {{WarpSize}}) {
      r = reduceOp(r, smem[i]);
    }

    smem[get_local_id(0)] = r;
  }

  // First thread will perform reductions across the block
  barrier(CLK_LOCAL_MEM_FENCE);

  float r = init;
  if (get_local_id(0) == 0) {
    r = smem[0];

    int numLanesParticipating = min(numVals, {{WarpSize}});

    if (numLanesParticipating == 32) {
      // Unroll for {{WarpSize}} == 32 and numVals >= 32
      // #pragma unroll
      // unrolling by hand, so compiler-independent
      {% for i=1,31 do %}
        r = reduceOp(r, smem[{{i}}]);
      {% end %}
    } else {
      for (int i = 1; i < numLanesParticipating; ++i) {
        r = reduceOp(r, smem[i]);
      }
    }
  }

  return r;
}


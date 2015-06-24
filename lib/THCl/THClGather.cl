// probably should put this on its own somewhere, so we 
// dont have to either ocpy/paste, or include entire THClReduceApplyUtils
typedef struct TensorInfoCl {
  unsigned int sizes[{{MAX_CLTORCH_DIMS}}];
  unsigned int strides[{{MAX_CLTORCH_DIMS}}];
  int offset;
  int dims;
} TensorInfoCl;

kernel void THClTensor_kernel_gather(
    global TensorInfoCl *dst_info, global float*dst_data,
    global const TensorInfoCl *src_info, global float*src_data,
   int dim,
    global const TensorInfoCl *idx_info, global float*idx_data,
   int totalElements
)
{
  for (int _linearId = get_global_id(0);
       _linearId < totalElements;
       _linearId += get_global_size(0)) {

      // plan is:
      // based on our linearIndex, this gets us a spot in the index
      // tensor
      // this is also a spot in the tgt_data (at least, if we can
      // convert into actual coordinates, then it is the coordinates
      // in the target tensor
      // the coordinates in the source are teh same, except that
      // we replace that of dimension dim with the value from
      // the index tensor
      //
      // so, everything hinges on us getting the coordinates, I think?
      // so, lets do that :-)
      int idxOffset = idx_info->offset;
      int srcOffset = src_info->offset;
      int dstOffset = dst_info->offset;
      int linearId = _linearId; // copy it, since we'll modify it
//      for(int d={{dims}}-1; d >= 0; d--) {  // just use slow, unbkaed loop for now, to
                                   // get it working
        int curDimIndex;
        {% for d=dims-1,0,-1 do %}
          curDimIndex = linearId % idx_info->sizes[{{d}}];
          idxOffset += curDimIndex * idx_info->strides[{{d}}];
          dstOffset += curDimIndex * dst_info->strides[{{d}}];
          if( {{d}} != dim ) { // this only matters for the source, the others are 
                           // unaffected by which dimension we are on. I think.
            srcOffset += curDimIndex * src_info->strides[{{d}}];
          }
          linearId /= idx_info->sizes[{{d}}];
        {% end %}
//      }
      // now we have the idxoffset.  get the value at that location
      int idxValue = idx_data[idxOffset] - 1; // subtract 1, because 1-based
      // then use this to get the final value for srcOffset
      srcOffset += idxValue * src_info->strides[dim];
      // get the value...
      float value = src_data[srcOffset];
      // and save it up...
      dst_data[dstOffset] = value;
      // thats it?
  }
}


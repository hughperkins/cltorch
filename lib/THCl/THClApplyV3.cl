// this will try to use float4s and stuff to retrieve from memory, in somewhat coallesced fashion
// (not quite aligned for now...)

// this will download blocks of 16 x 4 float4s to local memory, then apply on those
// thats the plan anyway :-)
//
// each workgroup will handle one block of 16x4 float4s, download those, and work on those
// so we need to know which dimensions those will be, and send those into this
// we can probably sort the dimensions and strides such that they are always in 
// the ok order, prior to starting the kernel, then our kernel is simpler (order
// of strides doesnt matter for apply, as long as all tensors are permuted in
// the same way)

typedef struct TensorInfoCl {
  unsigned int sizes[{{MAX_CLTORCH_DIMS}}]; // note: this is redundant between a/b
  unsigned int strides[{{MAX_CLTORCH_DIMS}}];
  int offset;
  int dims; //redundant
} TensorInfoCl;

void op( global float *out, global float *in1
  {% for i=1,(num_scalars) do %}
  , float val{{i}}
  {% end %}
) {
    {{operation}};
}

kernel void
THClTensor_pointwiseApply2( // let's specialize by hand...
    const int chunksPerPlane,
    global TensorInfoCl *a_info,
    global float*a_data,
    global TensorInfoCl *b_info,
    global float*b_data,
   {% for i=1,num_scalars do %}
   float val{{i}},
   {% end %}) {

  // first, we need to download our block into local memory
  // hmmm, each tensor only needs 1KB of local memory
  // two tensors is 2KB.  Not very much
  // anyway, let's ditch the outer for loop for now

  // within the last plane, the position of the chunk is (chunkX, chunkY)
  // where chunkX is along dim - 2, and chunkY is along dim - 1
  // each chunk is 16 x 16, always (but we might not load, or process,
  // some part of it)
  // the size of the plane, in chunks is planeSizeX, planeSizeY
  // we will process the plane by row, ie first two chunks will plausibyl
  // have same chunkY, and different chunkX

  int workgroupId = get_group_id(0);
//  int planeId = workgroupId / chunksPerPlane;
  int a_offset = a_info->offset;
  int b_offset = b_info->offset;
  int planeId = workgroupId / chunksPerPlane;
  int chunkXY = workgroupId % chunksPerPlane;
  int planeSizeX = (a_info->sizes[{{d}-2] + 16 - 1)/16;
  int planeSizeY = (a_info->sizes[{{d}-1] + 16 - 1)/16;
  int chunkY = chunkXY / planeSizeX;
  int chunkX = chunkXY % planeSizeX;
  int interPlaneOffset = 0;
  if({{nDims}} > 2) {
    interPlaneOffset = a_info->strides[{{nDims}} - 3];  
  }
  for(int d={{nDims}}-3; d >= 0; d--) {  // just use slow, unbkaed loop for now, to
                               // get it working
    int curDimIndex;
    curDimIndex = planeId % a_info->sizes[{{d}}];
    a_offset += curDimIndex * a_info->strides[{{d}}];
    srcOffset += curDimIndex * src_info->strides[{{d}}];
    if( {{d}} != dim ) { // this only matters for the source, the others are 
                     // unaffected by which dimension we are on. I think.
      dstOffset += curDimIndex * dst_info->strides[{{d}}];
    }
    linearId /= idx_info->sizes[{{d}}];
  }


  int linearId = get_global_id(0);
  // so, we have to figure out the coordinates of our block
  // need to loop over all dims, except hte last two
  // actually, we dont so much need the coordinates, as the 
  // offset into the new tensors
    int a_offset = a_info->offset;
    int b_offset = b_info->offset;
    // hmmm, note that we do in fact need to loop over all dims,
    // in order to find the start offset of our block
    for(int d={{nDims}}-1; d >= 0; d--) {  // just use slow, unbkaed loop for now, to
                                 // get it working
      int curDimIndex;
        curDimIndex = linearId % idx_info->sizes[{{d}}];
        idxOffset += curDimIndex * idx_info->strides[{{d}}];
        srcOffset += curDimIndex * src_info->strides[{{d}}];
        if( {{d}} != dim ) { // this only matters for the source, the others are 
                         // unaffected by which dimension we are on. I think.
          dstOffset += curDimIndex * dst_info->strides[{{d}}];
        }
        linearId /= idx_info->sizes[{{d}}];
    }

  {% for input_idx=1,num_tensors do %}
  // Convert `linearIndex` into an offset of `a`
  const int offset{{input_idx}} =
    IndexToOffset_{{1000+loadstring('return dim' .. input_idx)()}}_get(linearIndex, info_{{input_idx}}[0]);
  {% end %}

  op(a_data + a_offset, b_data + b_offset
    {% for i=1,num_scalars do %}
    , val{{i}}
    {% end %}
  );
}



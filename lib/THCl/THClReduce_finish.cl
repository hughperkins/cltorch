

void operation( float *out, *float in1 ) {
  {{operation}};
}

// concept this takes the start and end indices
// into the input array, reduces those with
// the zeroth index of the output array
// we use a few threads to retrieve the data
// and then one thread to reduce them, so only need
// one barrier ,but still get coallessced access
// N is number of values to reduce, starting and including
// startIndex
// assume only one workgroup (should be <= 15 values to get)
kernel void reduce( int startIndex, int N, global float *out, global float const*in, local float *_buffer)
{
  float reduced = out[0]; // request this now, so it's ready later
  // I suppose?
  int globalId = get_global_id(0);
  if( globalId >= N ) {
    return;
  }
  _buffer[i] = in[startIndex + globalId];
  // barrier...
  barrier(CLK_LOCAL_MEM_FENCE);
  if( globalId == 0 ) {
    for( int i = 0; i < N; i++ ) {
      operation( &reducedd, &(_data[i]));
    }
    out[0] = reduced;
  }
}



// this tidies up the values at the end
// it takes in an array of ints, which are the indices
// of the values to finish reducing
//
// concept: each thread fetches one value to local storage
// then one thread sums them, and writes to global memory
// we assume only one workgroup here (otherwise we'd 
// be uing the other kernel, right?)
//kernel void reduce(int N, global int *indices, global float *data, local float *_data) {
//  const int globalId = get_global_id(0);
//  if( globalId >= N ) {
//    return 0;
//  }
//  // this gives coallesced access I guess?
//  _data[globalId] = data[indices[globalId]];
//  // need one barrier, now
//  barrier(CLK_LOCAL_MEM_FENCE);
//  // poor last thread adds them up
//  if( globalId == 0 ) {
//    float value = _data[0];
//    for( int i = 1; i < N; i++ ) {
//      operation( &value, &(_data[i]) );
//    }
//    // save the result
//    data[0] = value;
//  }
//}




void operation( float *out, *float in1 ) {
  {{operation}};
}

// lets do this in pre-baked segments of 16
// we'll separate out any left-over values each time
// and process those at the end somehow
kernel void reduce(int numSegments, global float *out, global float *in) {
  const int globalId = get_global_id(0);
  const int segmentId = globalId;

  if( segmentId >= numSegments ) {
      return;
  }

  global const float *segment = in + segmentId * {{segmentLength}};
  float reduced = segment[0];
  
  // this could be done with #pragma too, but then
  // we have to check the doc for every vendor, to figure
  // out how to make sure it works ok.  This way, below,
  //  what you see is what you get :-)
  {% for i=1,segment_length do %}
    operation(&reduced, &(segment[i]));
  {% end %}
  out[segmentId] = reduced;
}


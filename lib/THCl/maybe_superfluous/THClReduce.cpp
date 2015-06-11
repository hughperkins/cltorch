#include <string>

#include "EasyCL.h"
#include "util/easycl_stringhelper.h"
#include "templates/TemplatedKernel.h"
#include "THClReduceApplyUtils.h"
#include "THClReduce.h"

using namespace std;

std::string THClReduce_get_bakedTemplate();
std::string THClReduce_get_finishTemplate();

float THClReduce_reduce(THClState *state, CLWrapper *in, HasOperator2 *op)
{
  const int segmentSize = 16;
  TemplatedKernel kernelBuilder(state->cl);
  kernelBuilder.set("segment_length", segmentSize);
  kernelBuilder.set("operation", op->operator2());
  CLKernel *baked = kernelBuilder.buildKernel(
    "THClReduce_baked_" + easycl::toString(segmentSize) + "_" + op->operator2(), "THClReduce_baked.cl",
    THClReduce_get_bakedTemplate(), "reduce" );
  CLKernel *finisher = kernelBuilder.buildKernel(
    "THClReduce_finish_" + op->operator2(), "THClReduce_finish.cl",
    THClReduce_get_finishTemplate(), "reduce" );
  int N = in->size();
  float *thisIn = 0;
  CLWrapper *thisInWrapper = in;
  CLWrapper *outWrapper = 0;
  float *out = 0;
  while( N > 1 ) {
    int numSegments = N / segmentSize;
    out = new float[numSegments];
    outWrapper = state->cl->wrap(numSegments, out);
    outWrapper->createOnDevice();
    int remainder = N % segmentSize;
    baked->in(numSegments)->out(outWrapper)->in(thisInWrapper);
    int workgroupSize = 64;
    int numWorkgroups = ( numSegments + workgroupSize - 1 ) / workgroupSize;
    baked->run_1d(numSegments * numWorkgroups, workgroupSize);
    state->cl->finish();
    if(remainder > 0) {
      finisher->in(N - remainder)->out(outWrapper)->in(thisInWrapper)->in(remainder);
      finisher->run_1d(numSegments, numSegments);
      state->cl->finish();
    }
    // now, the first numSegments values of thisOutWrapper need to be reduced
    N = numSegments;
    if( thisInWrapper != in ) {
      delete thisInWrapper;
      delete[] thisIn;
    }
    thisInWrapper = outWrapper;
    thisIn = out;
  }
  outWrapper->copyToHost();
  float result = out[0];
  delete outWrapper;
  delete[] out;
  return result;
}

std::string THClReduce_get_bakedTemplate()
{
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClReduce_baked.cl" )
  // ]]]
  // generated using cog, from THClReduce_baked.cl:
  const char * kernelSource =  
  "\n" 
  "\n" 
  "void operation( float *out, *float in1 ) {\n" 
  "  {{operation}};\n" 
  "}\n" 
  "\n" 
  "// lets do this in pre-baked segments of 16\n" 
  "// we'll separate out any left-over values each time\n" 
  "// and process those at the end somehow\n" 
  "kernel void reduce(int numSegments, global float *out, global float *in) {\n" 
  "  const int globalId = get_global_id(0);\n" 
  "  const int segmentId = globalId;\n" 
  "\n" 
  "  if( segmentId >= numSegments ) {\n" 
  "      return;\n" 
  "  }\n" 
  "\n" 
  "  global const float *segment = in + segmentId * {{segmentLength}};\n" 
  "  float reduced = segment[0];\n" 
  "\n" 
  "  // this could be done with #pragma too, but then\n" 
  "  // we have to check the doc for every vendor, to figure\n" 
  "  // out how to make sure it works ok.  This way, below,\n" 
  "  //  what you see is what you get :-)\n" 
  "  {% for i=1,segment_length do %}\n" 
  "    operation(&reduced, &(segment[i]));\n" 
  "  {% end %}\n" 
  "  out[segmentId] = reduced;\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}

std::string THClReduce_get_finishTemplate()
{
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClReduce_finish.cl" )
  // ]]]
  // generated using cog, from THClReduce_finish.cl:
  const char * kernelSource =  
  "\n" 
  "\n" 
  "void operation( float *out, *float in1 ) {\n" 
  "  {{operation}};\n" 
  "}\n" 
  "\n" 
  "// concept this takes the start and end indices\n" 
  "// into the input array, reduces those with\n" 
  "// the zeroth index of the output array\n" 
  "// we use a few threads to retrieve the data\n" 
  "// and then one thread to reduce them, so only need\n" 
  "// one barrier ,but still get coallessced access\n" 
  "// N is number of values to reduce, starting and including\n" 
  "// startIndex\n" 
  "// assume only one workgroup (should be <= 15 values to get)\n" 
  "kernel void reduce( int startIndex, int N, global float *out, global float const*in, local float *_buffer)\n" 
  "{\n" 
  "  float reduced = out[0]; // request this now, so it's ready later\n" 
  "  // I suppose?\n" 
  "  int globalId = get_global_id(0);\n" 
  "  if( globalId >= N ) {\n" 
  "    return;\n" 
  "  }\n" 
  "  _buffer[i] = in[startIndex + globalId];\n" 
  "  // barrier...\n" 
  "  barrier(CLK_LOCAL_MEM_FENCE);\n" 
  "  if( globalId == 0 ) {\n" 
  "    for( int i = 0; i < N; i++ ) {\n" 
  "      operation( &reducedd, &(_data[i]));\n" 
  "    }\n" 
  "    out[0] = reduced;\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "\n" 
  "\n" 
  "// this tidies up the values at the end\n" 
  "// it takes in an array of ints, which are the indices\n" 
  "// of the values to finish reducing\n" 
  "//\n" 
  "// concept: each thread fetches one value to local storage\n" 
  "// then one thread sums them, and writes to global memory\n" 
  "// we assume only one workgroup here (otherwise we'd\n" 
  "// be uing the other kernel, right?)\n" 
  "//kernel void reduce(int N, global int *indices, global float *data, local float *_data) {\n" 
  "//  const int globalId = get_global_id(0);\n" 
  "//  if( globalId >= N ) {\n" 
  "//    return 0;\n" 
  "//  }\n" 
  "//  // this gives coallesced access I guess?\n" 
  "//  _data[globalId] = data[indices[globalId]];\n" 
  "//  // need one barrier, now\n" 
  "//  barrier(CLK_LOCAL_MEM_FENCE);\n" 
  "//  // poor last thread adds them up\n" 
  "//  if( globalId == 0 ) {\n" 
  "//    float value = _data[0];\n" 
  "//    for( int i = 1; i < N; i++ ) {\n" 
  "//      operation( &value, &(_data[i]) );\n" 
  "//    }\n" 
  "//    // save the result\n" 
  "//    data[0] = value;\n" 
  "//  }\n" 
  "//}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}


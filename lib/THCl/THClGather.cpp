#include "THClTensorMath.h"
#include "THClGeneral.h"
//#include "THClBlas.h"
#include "THClTensorCopy.h"
//#include "THClTensorRandom.h"
#include "THClApply.h"
#include "THClReduce.h"
#include "THClKernels.h"
#include "THClReduceApplyUtils.h"

#include <iostream>
#include <string>
using namespace std;

//static const int maxClTorchDims = MAX_CL_TORCH_DIMS;

static std::string getTemplate();

//inline dim3 getApplyBlock(THClState *state) {
//  return dim3(getWorkgroupSize(state));
//}

//inline bool getApplyGrid(THClState* state, long totalElements, dim3& grid) {
////  int curDevice = -1;
////  cudaGetDevice(&curDevice);

////  if (curDevice == -1) {
////    return false;
////  }

////  // Assume a reasonable number of SMs if no state is available
////  int numSM =
////    state ? state->deviceProperties[curDevice].multiProcessorCount : 15;

//  // dont think we can get number of SMs in OpenCL? (at least, not in opencl 1.1?)
//  // just hardcode to 16 for now...
//  // FIXME
//  int numSM = 16;

//  // 16 warps per block * 4 per SM gives 64 warps per SM at maximum,
//  // which seems to be a good sweetspot for latency hiding
//  grid = dim3(mymin(DIVUP(totalElements, (long long) getWorkgroupSize(state)),
//                  4LL * numSM));
//  return true;
//}

THCL_API void THClTensor_gather(THClState *state, THClTensor *self, THClTensor *src, long dim, THClTensor *index) {
  // src will be ndims
  // index will be ndims too, though one of the dims should have length 1
  // self will be ndims
  int nDims = src->nDimension;
  cout << "nDims " << nDims << endl;

  THArgCheck(nDims >= 2, 2, "Tensors should have at least 2 dimensions"); // I guess?
//  THArgCheck(self->nDimension == nDims, 2, "All tensors should have same number of dims");
  THArgCheck(src->nDimension == nDims, 2, "All tensors should have same number of dims");
  THArgCheck(index->nDimension == nDims, 4, "All tensors should have same number of dims");
  THArgCheck(dim < nDims, 4, "dim out of bounds");
  THArgCheck(dim >= 0, 4, "dim out of bounds");
//  string message = 
//  int maxClTorchDims = MAX_CLTORCH_DIMS;
  THArgCheck(nDims < MAX_CLTORCH_DIMS, 2, "Tensors should have less than %i dimensions", MAX_CLTORCH_DIMS); // I guess?

  THLongStorage *newSize;

  for( int i = 0; i < nDims; i++ ) {
    if( i != dim ) {
      THArgCheck(THClTensor_size(state, src, i) == THClTensor_size(state, index, i), 3, ("index tensor must have same dimensions as source tensor, but dimension " + easycl::toString(i) + " doesnt match").c_str());
    }
  }

  newSize = THLongStorage_newWithSize(index->nDimension);
  THLongStorage_rawCopy(newSize, index->size);
//  newSize->data[dim] = nIndex;
  THClTensor_resize(state, self, newSize, NULL);
  THLongStorage_free(newSize);

  // This is just here to prove we are actually executing thi function :-)
  THClTensor_fill(state, self, 0);

  // since self is write-only, and index and src are read-only, ie none are read-write
  // so, we dnot need to worry about contiguity (at least, not from point of view of correctness)
  

  TemplatedKernel kernelBuilder( THClState_getCl(state) );
  kernelBuilder.set("IndexType", "int");
  kernelBuilder.set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS);
  std::string uniqueName = __FILE__ ":gather";
  CLKernel *kernel = kernelBuilder.buildKernel( uniqueName, __FILE__, getTemplate(), "THClTensor_kernel_gather" );

  TensorInfo<unsigned int> selfInfo(state, self);
    TensorInfo<unsigned int> srcInfo(state, src);
    TensorInfo<unsigned int> indexInfo(state, index);

  const dim3 block = getApplyBlock(state);

  long totalElements = THClTensor_nElement(state, index);
  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    THError("Couldnt create appropriate grid dimensions");
  }

  THClKernels k(state, kernel);
  k.out(selfInfo);
  k.in(srcInfo);
  k.in((int)dim);
  k.in(indexInfo);
  if( totalElements > ( 1l << 30 )) {
    throw std::runtime_error("Error: out of bounds for totalelements=" + easycl::toString(totalElements));
  }
  k.in( (int)totalElements );
  k.run(grid, block);

}

static std::string getTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClGather.cl" )
  // ]]]
  // generated using cog, from THClGather.cl:
  const char * kernelSource =  
  "// probably should put this on its own somewhere, so we\n" 
  "// dont have to either ocpy/paste, or include entire THClReduceApplyUtils\n" 
  "typedef struct TensorInfoCl {\n" 
  "  {{IndexType}} sizes[{{MAX_CLTORCH_DIMS}}];\n" 
  "  {{IndexType}} strides[{{MAX_CLTORCH_DIMS}}];\n" 
  "  {{IndexType}} offset;\n" 
  "  int dims;\n" 
  "} TensorInfoCl;\n" 
  "\n" 
  "kernel void THClTensor_kernel_gather(\n" 
  "    global TensorInfoCl *dst_info, global float*dst_data,\n" 
  "    global const TensorInfoCl *src_info, global float*src_data,\n" 
  "   int dim,\n" 
  "    global const TensorInfoCl *idx_info, global float*idx_data,\n" 
  "   int totalElements\n" 
  ")\n" 
  "{\n" 
  "//  global float *dst = dst_data + dst_info.offset;\n" 
  "//  global const float *src = src_data + src_info.offset;\n" 
  "//  global const float *idx = idx_data + idx_info.offset;\n" 
  "\n" 
  "  for (int _linearId = get_global_id(0);\n" 
  "       _linearId < totalElements;\n" 
  "       _linearId += get_global_size(0)) {\n" 
  "\n" 
  "      // plan is:\n" 
  "      // based on our linearIndex, this gets us a spot in the index\n" 
  "      // tensor\n" 
  "      // this is also a spot in the tgt_data (at least, if we can\n" 
  "      // convert into actual coordinates, then it is the coordinates\n" 
  "      // in the target tensor\n" 
  "      // the coordinates in the source are teh same, except that\n" 
  "      // we replace that of dimension dim with the value from\n" 
  "      // the index tensor\n" 
  "      //\n" 
  "      // so, everything hinges on us getting the coordinates, I think?\n" 
  "      // so, lets do that :-)\n" 
  "      int idxOffset = idx_info->offset;\n" 
  "      int srcOffset = src_info->offset;\n" 
  "      int dstOffset = dst_info->offset;\n" 
  "      int linearId = _linearId; // copy it, since we'll modify it\n" 
  "      for(int d=dim-1; d >= 0; d--) {  // just use slow, unbkaed loop for now, to\n" 
  "                                   // get it working\n" 
  "        int curDimIndex = linearId % idx_info->sizes[d];\n" 
  "        idxOffset += curDimIndex * idx_info->strides[d];\n" 
  "        dstOffset += curDimIndex * dst_info->strides[d];\n" 
  "        if( d != dim ) { // this only matters for the source, the others are\n" 
  "                         // unaffected by which dimension we are on. I think.\n" 
  "          srcOffset += curDimIndex * src_info->strides[d];\n" 
  "        } else {\n" 
  "          // do nothing... add it later, once we know the value\n" 
  "        }\n" 
  "        linearId /= idx_info->sizes[d];\n" 
  "      }\n" 
  "      // now we have the idxoffset.  get the value at that location\n" 
  "      int idxValue = idx_data[idxOffset];\n" 
  "      // then use this to get the final value for srcOffset\n" 
  "      srcOffset += idxValue * src_info->strides[dim];\n" 
  "      // get the value...\n" 
  "      float value = src_data[srcOffset];\n" 
  "      // and save it up...\n" 
  "      dst_data[dstOffset] = value;\n" 
  "      // thats it?\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}


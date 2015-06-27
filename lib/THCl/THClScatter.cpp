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

static std::string getTemplate();

THCL_API void THClTensor_scatter(THClState *state, THClTensor *self, long dim, THClTensor *index, THClTensor *src) {
  int nDims = index->nDimension;

  THArgCheck(nDims >= 2, 2, "Tensors should have at least 2 dimensions"); // I guess?
  THArgCheck(src->nDimension == nDims, 4, "All tensors should have same number of dims");
  THArgCheck(index->nDimension == nDims, 3, "All tensors should have same number of dims");
  THArgCheck(dim < nDims, 2, "dim out of bounds");
  THArgCheck(dim >= 0, 2, "dim out of bounds");
  THArgCheck(nDims < MAX_CLTORCH_DIMS, 2, "Tensors should have less than %i dimensions", MAX_CLTORCH_DIMS); // I guess?

//  THLongStorage *newSize;

  for( int i = 0; i < nDims; i++ ) {
    if( i != dim ) {
      THArgCheck(THClTensor_size(state, src, i) == THClTensor_size(state, index, i), 3, ("index tensor must have same dimensions as source tensor, but dimension " + easycl::toString(i) + " doesnt match").c_str());
    }
  }

//  if( self != src ) {
//    newSize = THLongStorage_newWithSize(index->nDimension);
//    THLongStorage_rawCopy(newSize, index->size);
//    THClTensor_resize(state, self, newSize, NULL);
//    THLongStorage_free(newSize);
//  }

  // since self is write-only, and index and src are read-only, ie none are read-write
  // so, we dnot need to worry about contiguity (at least, not from point of view of correctness)

 // hmmm ,we should probably check that self is not aliased
  THArgCheck(self != index, 1, "self cannot alias index");
  THArgCheck(self != src, 1, "self cannot alias src");
  
  std::string uniqueName = __FILE__ ":scatter:" + easycl::toString(nDims);
  EasyCL *cl = THClState_getCl(state);
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("Apply3 1aa");
  } else {
    TemplatedKernel kernelBuilder( THClState_getCl(state) );
    kernelBuilder.set("IndexType", "unsigned int");
    kernelBuilder.set("dims", nDims);
    kernelBuilder.set("scatter", 1);
    kernelBuilder.set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS);
    kernel = kernelBuilder.buildKernel( uniqueName, __FILE__, getTemplate(), "THClTensor_kernel_scatter" );
  }

  TensorInfoCl selfInfoCl(self);
    TensorInfoCl srcInfoCl(src);
    TensorInfoCl indexInfoCl(index);

  const dim3 block = getApplyBlock(state);

  long totalElements = THClTensor_nElement(state, index);
  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    THError("Couldnt create appropriate grid dimensions");
  }

  THClKernels k(state, kernel);
  kernel->in(1, &selfInfoCl);
  kernel->out(self->storage->wrapper);
  k.in((int)dim);
  kernel->in(1, &indexInfoCl);
  kernel->in(index->storage->wrapper);
  kernel->in(1, &srcInfoCl);
  kernel->in(src->storage->wrapper);
  if( totalElements > ( 1l << 30 )) {
    throw std::runtime_error("Error: out of bounds for totalelements=" + easycl::toString(totalElements));
  }
  k.in( (int)totalElements );
  k.run(grid, block);

}


THCL_API void THClTensor_scatterFill(THClState *state, THClTensor *self, long dim, THClTensor *index, float src_val) {
  int nDims = index->nDimension;

  THArgCheck(nDims >= 2, 2, "Tensors should have at least 2 dimensions"); // I guess?
  THArgCheck(index->nDimension == nDims, 3, "All tensors should have same number of dims");
  THArgCheck(dim < nDims, 2, "dim out of bounds");
  THArgCheck(dim >= 0, 2, "dim out of bounds");
  THArgCheck(nDims < MAX_CLTORCH_DIMS, 2, "Tensors should have less than %i dimensions", MAX_CLTORCH_DIMS); // I guess?

  for( int i = 0; i < nDims; i++ ) {
    if( i != dim ) {
      THArgCheck(THClTensor_size(state, self, i) == THClTensor_size(state, index, i), 3, ("index tensor must have same dimensions as destination tensor, but dimension " + easycl::toString(i) + " doesnt match").c_str());
    }
  }

  // since self is write-only, and index and src are read-only, ie none are read-write
  // so, we dnot need to worry about contiguity (at least, not from point of view of correctness)
  THArgCheck(self != index, 1, "self cannot alias index");
  
  std::string uniqueName = __FILE__ ":scatterFill:" + easycl::toString(nDims);
  EasyCL *cl = THClState_getCl(state);
  CLKernel *kernel = 0;
  if(cl->kernelExists(uniqueName)) {
    kernel = cl->getKernel(uniqueName);
    StatefulTimer::timeCheck("Apply3 1aa");
  } else {
    TemplatedKernel kernelBuilder( THClState_getCl(state) );
    kernelBuilder.set("IndexType", "unsigned int");
    kernelBuilder.set("dims", nDims);
    kernelBuilder.set("scatterFill", 1);
    kernelBuilder.set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS);
    kernel = kernelBuilder.buildKernel( uniqueName, __FILE__, getTemplate(), "THClTensor_kernel_scatterFill" );
  }

  TensorInfoCl selfInfoCl(self);
    TensorInfoCl indexInfoCl(index);

  const dim3 block = getApplyBlock(state);

  long totalElements = THClTensor_nElement(state, index);
  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    THError("Couldnt create appropriate grid dimensions");
  }

  THClKernels k(state, kernel);
  kernel->in(1, &selfInfoCl);
  kernel->out(self->storage->wrapper);
  k.in((int)dim);
  kernel->in(1, &indexInfoCl);
  kernel->in(index->storage->wrapper);
  k.in(src_val);
  if( totalElements > ( 1l << 30 )) {
    throw std::runtime_error("Error: out of bounds for totalelements=" + easycl::toString(totalElements));
  }
  k.in( (int)totalElements );
  k.run(grid, block);

}

static std::string getTemplate() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClScatter.cl" )
  // ]]]
  // generated using cog, from THClScatter.cl:
  const char * kernelSource =  
  "// probably should put this on its own somewhere, so we\n" 
  "// dont have to either ocpy/paste, or include entire THClReduceApplyUtils\n" 
  "typedef struct TensorInfoCl {\n" 
  "  unsigned int sizes[{{MAX_CLTORCH_DIMS}}];\n" 
  "  unsigned int strides[{{MAX_CLTORCH_DIMS}}];\n" 
  "  int offset;\n" 
  "  int dims;\n" 
  "} TensorInfoCl;\n" 
  "\n" 
  "{% if scatter then %}\n" 
  "kernel void THClTensor_kernel_scatter(\n" 
  "    global TensorInfoCl *dst_info, global float*dst_data,\n" 
  "   int dim,\n" 
  "    global const TensorInfoCl *idx_info, global float*idx_data,\n" 
  "    global const TensorInfoCl *src_info, global float*src_data,\n" 
  "   int totalElements\n" 
  ")\n" 
  "{\n" 
  "  for (int _linearId = get_global_id(0);\n" 
  "       _linearId < totalElements;\n" 
  "       _linearId += get_global_size(0)) {\n" 
  "\n" 
  "      // plan is:\n" 
  "      // based on our linearIndex, this gets us a spot in the index\n" 
  "      // tensor\n" 
  "      // this is also a spot in the src_data (at least, if we can\n" 
  "      // convert into actual coordinates, then it is the coordinates\n" 
  "      // in the src tensor\n" 
  "      // the coordinates in the dest are teh same, except that\n" 
  "      // we replace that of dimension dim with the value from\n" 
  "      // the index tensor\n" 
  "      //\n" 
  "      // so, everything hinges on us getting the coordinates, I think?\n" 
  "      // so, lets do that :-)\n" 
  "      int idxOffset = idx_info->offset;\n" 
  "      int srcOffset = src_info->offset;\n" 
  "      int dstOffset = dst_info->offset;\n" 
  "      int linearId = _linearId; // copy it, since we'll modify it\n" 
  "//      for(int d={{dims}}-1; d >= 0; d--) {  // just use slow, unbkaed loop for now, to\n" 
  "                                   // get it working\n" 
  "        int curDimIndex;\n" 
  "        {% for d=dims-1,0,-1 do %}\n" 
  "          curDimIndex = linearId % idx_info->sizes[{{d}}];\n" 
  "          idxOffset += curDimIndex * idx_info->strides[{{d}}];\n" 
  "          srcOffset += curDimIndex * src_info->strides[{{d}}];\n" 
  "          if( {{d}} != dim ) { // this only matters for the source, the others are\n" 
  "                           // unaffected by which dimension we are on. I think.\n" 
  "            dstOffset += curDimIndex * dst_info->strides[{{d}}];\n" 
  "          }\n" 
  "          linearId /= idx_info->sizes[{{d}}];\n" 
  "        {% end %}\n" 
  "//      }\n" 
  "      // now we have the idxoffset.  get the value at that location\n" 
  "      int idxValue = idx_data[idxOffset] - 1; // subtract 1, because 1-based\n" 
  "      // then use this to get the final value for dstOffset\n" 
  "      dstOffset += idxValue * dst_info->strides[dim];\n" 
  "      // get the value...\n" 
  "      float value = src_data[srcOffset];\n" 
  "      // and save it up...\n" 
  "      dst_data[dstOffset] = value;\n" 
  "      // thats it?\n" 
  "  }\n" 
  "}\n" 
  "{% end %}\n" 
  "\n" 
  "{% if scatterFill then %}\n" 
  "kernel void THClTensor_kernel_scatterFill(\n" 
  "    global TensorInfoCl *dst_info, global float*dst_data,\n" 
  "   const int dim,\n" 
  "    global const TensorInfoCl *idx_info, global float*idx_data,\n" 
  "    const float src_val,\n" 
  "   const int totalElements\n" 
  ")\n" 
  "{\n" 
  "  for (int _linearId = get_global_id(0);\n" 
  "       _linearId < totalElements;\n" 
  "       _linearId += get_global_size(0)) {\n" 
  "\n" 
  "      // plan is:\n" 
  "      // based on our linearIndex, this gets us a spot in the index\n" 
  "      // tensor\n" 
  "      // the coordinates in the dest are teh same, except that\n" 
  "      // we replace that of dimension dim with the value from\n" 
  "      // the index tensor\n" 
  "      //\n" 
  "      // so, everything hinges on us getting the coordinates, I think?\n" 
  "      // so, lets do that :-)\n" 
  "      int idxOffset = idx_info->offset;\n" 
  "      int dstOffset = dst_info->offset;\n" 
  "      int linearId = _linearId; // copy it, since we'll modify it\n" 
  "//      for(int d={{dims}}-1; d >= 0; d--) {  // just use slow, unbkaed loop for now, to\n" 
  "                                   // get it working\n" 
  "        int curDimIndex;\n" 
  "        {% for d=dims-1,0,-1 do %}\n" 
  "          curDimIndex = linearId % idx_info->sizes[{{d}}];\n" 
  "          idxOffset += curDimIndex * idx_info->strides[{{d}}];\n" 
  "          if( {{d}} != dim ) { // this only matters for the source, the others are\n" 
  "                           // unaffected by which dimension we are on. I think.\n" 
  "            dstOffset += curDimIndex * dst_info->strides[{{d}}];\n" 
  "          }\n" 
  "          linearId /= idx_info->sizes[{{d}}];\n" 
  "        {% end %}\n" 
  "//      }\n" 
  "      // now we have the idxoffset.  get the value at that location\n" 
  "      int idxValue = idx_data[idxOffset] - 1; // subtract 1, because 1-based\n" 
  "      // then use this to get the final value for dstOffset\n" 
  "      dstOffset += idxValue * dst_info->strides[dim];\n" 
  "      // and save value up...\n" 
  "      dst_data[dstOffset] = src_val;\n" 
  "      // thats it?\n" 
  "  }\n" 
  "}\n" 
  "{% end %}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}


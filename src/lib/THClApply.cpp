#include "THClApply.h"
#include "THClKernels.h"
#include "THClTypeParseTraits.h"
#include "EasyCL.h"
#include "CLKernel_structs.h"
#include "util/easycl_stringhelper.h"
#include "util/StatefulTimer.h"
#include "templates/TemplatedKernel.h"

#include <string>
#include <iostream>

using namespace std;

static std::string get_template();

// Called when we are copying into an overlapping index `dst`, but
// we don't care which writer wins. Hacky but it works.
void THClTensor_copyIgnoringOverlaps(THClState* state,
                                     THClTensor* dst,
                                     THClTensor* src) {
  CopyOp op;
  THClTensor_pointwiseApply2(state, dst, src, &op,
                               ReadOnly, // ignore overwrites
                               ReadOnly);
}

// Threads per block for our apply kernel
#define THCL_APPLY_THREADS_PER_BLOCK 32 * 16

int getWorkgroupSize(THClState *state, int device) {
  return 64;
}
dim3 getApplyBlock(THClState *state, int device) {
  return dim3(getWorkgroupSize(state, device));
}
dim3 getApplyGrid(THClState* state, int device, long totalElements) {
  int workgroupSize = getWorkgroupSize(state, device);
  return dim3((totalElements + workgroupSize - 1 ) / workgroupSize);
}

template< typename IndexType >
void kernelLaunch_pointwiseApply( THClState *state, dim3 grid, dim3 block, int numTensors, int *dims, TensorInfo<IndexType> **infos, IndexType totalElements, OpBase const*op, string operationString) {
  int numScalars = 0;
  HasScalars const*hasScalars = dynamic_cast<HasScalars const*>(op);
  if( hasScalars != 0 ) {
    numScalars = hasScalars->getNumScalars();
  }
  int numPointTensors = 0;
  HasPointTensors const*hasPointTensors = dynamic_cast<HasPointTensors const *>(op);
  if(hasPointTensors != 0) {
    numPointTensors = hasPointTensors->getNumPointTensors();
  }
  StatefulTimer::timeCheck("Apply getname");
  ostringstream oss;
  oss << "Apply_" << numTensors << "t_" << numScalars << "s_" << numPointTensors << "pt_";
  for(int t=0; t < numTensors; t++) {
    oss << dims[t] << "_";
  }
  oss << operationString;
  StatefulTimer::timeCheck("Apply gotname");
  if(false && StatefulTimer::enabled) {
    for(int t=0; t < numTensors; t++) {
      TensorInfo<IndexType> *info = infos[t];
      oss << "tensor " << t << ": ";
      oss << " dims=" << info->dims;
      oss << " sizes={";
      for(int d=0; d < info->dims; d++) {
        if(d > 0) {
          oss << ",";
        }
        oss << info->sizes[d];
      }
      oss << "}";
      oss << " strides={";
      for(int d=0; d < info->dims; d++) {
        if(d > 0) {
          oss << ",";
        }
        oss << info->strides[d];
      }
      oss << "}";
    }
    oss << " nelements=" << totalElements;
//    uniqueName = oss.str();
  }
  EasyCL *cl = infos[numTensors - 1]->wrapper->getCl();
  CLKernel *kernel = 0;
  if(cl->kernelExists(oss.str())) {
    kernel = cl->getKernel(oss.str());
  } else {
    string uniqueName = oss.str();
    TemplatedKernel kernelBuilder(cl);
    for(int t=0; t < numTensors; t++) {
      kernelBuilder.set("dims" + easycl::toString(t + 1), dims[t]);
    }
    kernelBuilder.set("num_tensors", numTensors);
    kernelBuilder.set("num_scalars", numScalars);
    kernelBuilder.set("num_point_tensors", numPointTensors);
    kernelBuilder.set("IndexType", TypeParseTraits<IndexType>::name);
    kernelBuilder.set("operation", operationString);
    kernel = kernelBuilder.buildKernel( uniqueName, uniqueName, get_template(), "THClTensor_pointwiseApplyD" );
//    cout << kernelBuilder.getRenderedKernel(get_template()) << endl;
    StatefulTimer::timeCheck("Apply compiled");
  }
  StatefulTimer::timeCheck("Apply got kernel");

  THClKernels k(state, kernel);

  for(int t=0; t < numTensors; t++) {
    TensorInfo<IndexType> *info = infos[t];
    k.in((int)info->offset);
    for(int d=0; d < dims[t]; d++ ) {
      k.in((int)info->sizes[d]);
      k.in((int)info->strides[d]);
    }
    if(t == 0) {
      k.inout(info->wrapper);
    } else {
      k.in(info->wrapper);
    }
  }
  for( int i = 0; i < numScalars; i++ ) {
    k.in(hasScalars->getScalar(i));
  }
  for( int i = 0; i < numPointTensors; i++ ) {
    k.in(hasPointTensors->getPointTensor(i)->storage->wrapper);
  }
  if( totalElements > ( 1l << 30 )) {
    throw std::runtime_error("Error: out of bounds for totalelements=" + easycl::toString(totalElements));
  }
  k.in( (int)totalElements );
  k.run(grid, block);

  if(state->addFinish) cl->finish();
  if(StatefulTimer::enabled) StatefulTimer::timeCheck(("Apply END " + oss.str()).c_str());
}
bool THClTensor_pointwiseApply(THClState* state,
                               int numTensors,
                               THClTensor** tensors,
                               const OpBase *op,
                               string operationString,
                               TensorArgType *types) {
  long totalElements = THClTensor_nElement(state, tensors[0]);
  const int device = tensors[0]->storage->device;
  for(int t=1; t < numTensors; t++) {
    if (totalElements != THClTensor_nElement(state, tensors[t])) {
      THError("element size mismatch between tensors 1 and %i", t+1);
      return false;
    }
  }
  for(int t=0; t < numTensors; t++) {
    if (THClTensor_nDimension(state, tensors[t]) > MAX_CLTORCH_DIMS) {
      THError("Apply: too many dimensions in tensor %i", (t+1));
      return false;
    }
  }

  if (THClTensor_nDimension(state, tensors[0]) == -1) {
    return true;
  }

  const dim3 block = getApplyBlock(state, device);
  const dim3 grid = getApplyGrid(state, device, totalElements);

  // If tensor args have overlapping indices and are read/write, then
  // we must expand the tensor to a contiguous form first, since
  // otherwise there are conflicting writes. Upon copying back to the
  // non-contiguous form, there will be conflicting writes, but at
  // least with copy, one of the updaters will win atomically. This is
  // a sketchy property of the old system as well (writing into all
  // indices of a tensor with overlapping indices should probably be
  // an error, since it is unclear which one should win), but we will
  // preserve this last-writer-wins (in arbitrary copy order) behavior.
  THClTensor *oldTensors[numTensors];
  memset(oldTensors, 0, sizeof(THClTensor *) * numTensors);
  for(int t=0; t < numTensors; t++) {
    if (types[t] == ReadWrite && THCL_overlappingIndices(state, tensors[t])) {
      oldTensors[t] = tensors[t];
      tensors[t] = THClTensor_newContiguous(state, tensors[t]);
    }
  }

  bool canUse32BitIndexMath = true;
  for(int t=0; t < numTensors; t++) {
    if(!THCL_canUse32BitIndexMath(state, tensors[t])) {
      canUse32BitIndexMath = false;
    }
  }
  if(canUse32BitIndexMath) {
    TensorInfo<uint32> *infos[numTensors];
    for(int t=0; t < numTensors; t++) {
      infos[t] = new TensorInfo<uint32>(state, tensors[t]);
    }
    int dims[numTensors];
    for(int t=0; t < numTensors; t++) {
      dims[t] = infos[t]->dims;
      if(infos[t]->isContiguous()) dims[t] = -2;
    }
    kernelLaunch_pointwiseApply< uint32 >(state, grid, block, numTensors, dims, infos, (unsigned int) totalElements, op, operationString);
    for(int t=0; t < numTensors; t++) {
      delete infos[t];
    }
  } else {
    TensorInfo<uint64> *infos[numTensors];
    bool allContiguous = true;
    for(int t=0; t < numTensors; t++) {
      infos[t] = new TensorInfo<uint64>(state, tensors[t]);
      if(!infos[t]->isContiguous()) {
        allContiguous = false;
      }
    }
    if (allContiguous) {
      THError("Not implemented");
    } else {
      THError("Not implemented");
    }
    for(int t=0; t < numTensors; t++) {
      delete infos[t];
    }
  }

  for(int t=0; t < numTensors; t++) {
    if (oldTensors[t]) {
      THClTensor_copyIgnoringOverlaps(state, oldTensors[t], tensors[t]);
      THClTensor_free(state, tensors[t]);
      tensors[t] = oldTensors[t];
    }
  }
  return true;
}
bool THClTensor_pointwiseApply1(THClState* state,
                                  THClTensor* a,
                                  const HasOperator1 *op,
                                  TensorArgType aType) {
  THClTensor *tensors[1];
  tensors[0] = a;
  TensorArgType types[1];
  types[0] = aType;
  return THClTensor_pointwiseApply(
    state, 1, tensors, op, op->operator1(), types);
}

bool THClTensor_pointwiseApply2(THClState* state,
                                  THClTensor* a,
                                  THClTensor* b,
                                  const HasOperator2 *op,
                                  TensorArgType aType,
                                  TensorArgType bType) {
  THClTensor *tensors[2];
  tensors[0] = a;
  tensors[1] = b;
  TensorArgType types[2];
  types[0] = aType;
  types[1] = bType;
  return THClTensor_pointwiseApply(
    state, 2, tensors, op, op->operator2(), types);
}

bool THClTensor_pointwiseApply3(THClState* state,
                                  THClTensor* a,
                                  THClTensor* b,
                                  THClTensor* c,
                                  const HasOperator3 *op,
                                  TensorArgType aType,
                                  TensorArgType bType,
                                  TensorArgType cType) {
  THClTensor *tensors[3];
  tensors[0] = a;
  tensors[1] = b;
  tensors[2] = c;
  TensorArgType types[3];
  types[0] = aType;
  types[1] = bType;
  types[2] = cType;
  return THClTensor_pointwiseApply(
    state, 3, tensors, op, op->operator3(), types);
}

std::string get_template() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "THClApply.cl" )
  // ]]]
  // generated using cog, from THClApply.cl:
  const char * kernelSource =  
  "// OpenCL kernels....\n" 
  "\n" 
  "// expected templated values:\n" 
  "// dims (vector of unique dimension values)\n" 
  "// operation\n" 
  "// dim1\n" 
  "// dim2\n" 
  "// dim3\n" 
  "// ... dimD\n" 
  "// num_input_tensors\n" 
  "// include_scalar_input\n" 
  "//\n" 
  "// maybe should add:\n" 
  "// IndexType (hardcoded to int for now)\n" 
  "// MAX_CUTORCH_DIMS (hardcoded to 25 for now)\n" 
  "\n" 
  "// (Ported from cutorch's THCApply.cuh)\n" 
  "\n" 
  "// Maximum number of dimensions allowed for cutorch\n" 
  "// #define MAX_CUTORCH_DIMS 25\n" 
  "\n" 
  "// Enum that indicates whether tensor arguments are read/write or\n" 
  "// read-only\n" 
  "//enum TensorArgType { ReadWrite, ReadOnly };\n" 
  "\n" 
  "{%\n" 
  " local total_opsize = num_tensors\n" 
  " if include_scalar_input then\n" 
  "      total_opsize = total_opsize + 1\n" 
  "   end\n" 
  " %}\n" 
  "\n" 
  "inline void op( global float *out\n" 
  "  {% for t=1,(num_tensors-1) do %}\n" 
  "  , global float *in{{t}}\n" 
  "  {% end %}\n" 
  "  {% for s=1,(num_scalars) do %}\n" 
  "  , float val{{s}}\n" 
  "  {% end %}\n" 
  "   {% for pt=1,num_point_tensors do %}\n" 
  "   , global float *pointTensor{{pt}}\n" 
  "   {% end %}\n" 
  ") {\n" 
  "    {{operation}};\n" 
  "}\n" 
  "\n" 
  "kernel void\n" 
  "THClTensor_pointwiseApplyD(\n" 
  "   {% for t=1,num_tensors do %}\n" 
  "    int offset_{{t}},\n" 
  "    {% local thisdims = loadstring('return dims' .. t)() %}\n" 
  "    {% for d=1,thisdims do %}\n" 
  "      int size_{{t}}_{{d}},\n" 
  "      int stride_{{t}}_{{d}},\n" 
  "    {% end %}\n" 
  "    global float*data_{{t}},\n" 
  "   {% end %}\n" 
  "   {% for i=1,num_scalars do %}\n" 
  "   float val{{i}},\n" 
  "   {% end %}\n" 
  "   {% for i=1,num_point_tensors do %}\n" 
  "   global float *pointTensor{{i}},\n" 
  "   {% end %}\n" 
  "   int totalElements) {\n" 
  "   int linearIndex = get_global_id(0);\n" 
  "   if(linearIndex < totalElements ) {\n" 
  "      int thisLinearId;\n" 
  "    {% for t=1,num_tensors do %}\n" 
  "      {% local thisdims = loadstring('return dims' .. t)() %}\n" 
  "      {% if thisdims == -2 then %}\n" 
  "         int derived_offset_{{t}} = linearIndex + offset_{{t}};\n" 
  "      {% else %}\n" 
  "         {{IndexType}} derived_offset_{{t}} = offset_{{t}};\n" 
  "         thisLinearId = linearIndex;\n" 
  "        {% for d=thisdims,1,-1 do %}  // bake this in....\n" 
  "          derived_offset_{{t}} += (thisLinearId % size_{{t}}_{{d}}) * stride_{{t}}_{{d}};\n" 
  "          {% if d > 0 then %}\n" 
  "            thisLinearId /= size_{{t}}_{{d}};\n" 
  "          {% end %}\n" 
  "        {% end %}\n" 
  "\n" 
  "      {% end %}\n" 
  "    {% end %}\n" 
  "\n" 
  "    op(\n" 
  "      {% for t=1,num_tensors do %}\n" 
  "         {% if t > 1 then %} , {% end %}\n" 
  "         &(data_{{t}}[derived_offset_{{t}}])\n" 
  "      {% end %}\n" 
  "\n" 
  "      {% for s=1,num_scalars do %}\n" 
  "      , val{{s}}\n" 
  "      {% end %}\n" 
  "\n" 
  "       {% for pt=1,num_point_tensors do %}\n" 
  "       , pointTensor{{pt}}\n" 
  "       {% end %}\n" 
  "    );\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}


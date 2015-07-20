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
// Implementation of copyIgnoringOverlaps, defined after pointwiseApply2.
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

// Called when we are copying into an overlapping index `dst`, but
// we don't care which writer wins. Hacky but it works.
void THClTensor_copyIgnoringOverlaps(THClState* state,
                                       THClTensor* dst,
                                       THClTensor* src);

int getWorkgroupSize(THClState *state, int device) {
  return 64;

//  int workgroupSize = THCL_APPLY_THREADS_PER_BLOCK;
//  int maxWorkgroupSize = ((easycl::DeviceInfo *)state->deviceInfoByDevice[device])->maxWorkGroupSize;
//  if( workgroupSize > maxWorkgroupSize ) {
//    workgroupSize = maxWorkgroupSize;
//  }
//  return workgroupSize;
}

dim3 getApplyBlock(THClState *state, int device) {
  return dim3(getWorkgroupSize(state, device));
}

bool getApplyGrid(THClState* state, int device, long totalElements, dim3& grid) {
//  int curDevice = -1;
//  cudaGetDevice(&curDevice);

//  if (curDevice == -1) {
//    return false;
//  }

//  // Assume a reasonable number of SMs if no state is available
//  int numSM =
//    state ? state->deviceProperties[curDevice].multiProcessorCount : 15;

  // dont think we can get number of SMs in OpenCL? (at least, not in opencl 1.1?)
  // just hardcode to 16 for now...
  // FIXME
//  int numSM = 16;

//  // 16 warps per block * 4 per SM gives 64 warps per SM at maximum,
//  // which seems to be a good sweetspot for latency hiding
//  grid = dim3(mymin(DIVUP(totalElements, (long long) getWorkgroupSize(state, device)),
//                  4LL * numSM));
  int workgroupSize = getWorkgroupSize(state, device);
  grid = dim3((totalElements + workgroupSize - 1 ) / workgroupSize);

  return true;
}

template< typename IndexType >
void kernelLaunch_pointwiseApply1( THClState *state, dim3 grid, dim3 block, int A, TensorInfo<IndexType> aInfo, IndexType totalElements, HasOperator1 const * op ) {
  if(StatefulTimer::enabled) StatefulTimer::timeCheck("Apply1 start");
  int numTensors = 1;
  int numScalars = 0;
  HasScalars const*hasScalars = dynamic_cast<HasScalars const*>(op);
  if(hasScalars != 0) {
    numScalars = hasScalars->getNumScalars();
  }
  int numPointTensors = 0;
  HasPointTensors const*hasPointTensors = dynamic_cast<HasPointTensors const *>(op);
  if(hasPointTensors != 0) {
    numPointTensors = hasPointTensors->getNumPointTensors();
  }
  ostringstream oss;
  oss << "Apply1t_" << numScalars << "s_" << numPointTensors << "pt_" << A << "_" << op->operator1();
  EasyCL *cl = aInfo.wrapper->getCl();
  CLKernel *kernel = 0;
  if( cl->kernelExists(oss.str()) ) {
    kernel = cl->getKernel(oss.str());
  } else {
    string uniqueName = oss.str();
    TemplatedKernel kernelBuilder(cl);
    kernelBuilder.set("dim1", A);
    std::vector<int> dims;
    if( A >= 0 ) {
      dims.push_back(A);
    }
    std::string operation = op->operator1();
    kernelBuilder.set("num_tensors", numTensors);
    kernelBuilder.set("num_scalars", numScalars);
    kernelBuilder.set("dims", dims);
    kernelBuilder.set("num_tensor_inputs", numTensors);
    kernelBuilder.set("num_point_tensors", numPointTensors);
    kernelBuilder.set("IndexType", TypeParseTraits<IndexType>::name);
    kernelBuilder.set("WarpSize", 32);
    kernelBuilder.set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS);
    kernelBuilder.set("operation", operation);
    kernelBuilder.set("include_THClReduceApplyUtils", THClReduceApplyUtils_getKernelTemplate());
    kernel = kernelBuilder.buildKernel( uniqueName, "THClApply.cl", get_template(), "THClTensor_pointwiseApplyD" );
      StatefulTimer::timeCheck("Apply1 compiled");
  }
  THClKernels k(state, kernel);
  k.out(aInfo);
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
  if(StatefulTimer::enabled) StatefulTimer::timeCheck("Apply1 END");
}

template< typename IndexType >
void kernelLaunch_pointwiseApply2( THClState *state, dim3 grid, dim3 block, int A, int B, TensorInfo<IndexType> aInfo, TensorInfo<IndexType> bInfo, IndexType totalElements, HasOperator2 const*op ) {
  StatefulTimer::timeCheck("Apply2 START");
  int numTensors = 2;
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
  ostringstream oss;
  oss << "Apply2t_" << numScalars << "s_" << numPointTensors << "pt_" << A << "_" << B << "_" << op->operator2();
//  std::string uniqueName = "THClApply_" + easycl::toString(numTensors) + "t" + easycl::toString(numScalars) + "s_" + easycl::toString(numPointTensors) + "pt_" + easycl::toString(A) + "_" + easycl::toString(B) + "_" + op->operator2();
  if(false && StatefulTimer::enabled) {
//    ostringstream oss;
    oss << " dims=" << aInfo.dims;
    oss << "tensor a: ";
    oss << " sizes={";
    for(int d=0; d < aInfo.dims; d++) {
      if(d > 0) {
        oss << ",";
      }
      oss << aInfo.sizes[d];
    }
    oss << "}";
    oss << " strides={";
    for(int d=0; d < aInfo.dims; d++) {
      if(d > 0) {
        oss << ",";
      }
      oss << aInfo.strides[d];
    }
    oss << "}";
    oss << "tensor b: ";
    oss << " sizes={";
    for(int d=0; d < bInfo.dims; d++) {
      if(d > 0) {
        oss << ",";
      }
      oss << bInfo.sizes[d];
    }
    oss << "}";
    oss << " strides={";
    for(int d=0; d < bInfo.dims; d++) {
      if(d > 0) {
        oss << ",";
      }
      oss << bInfo.strides[d];
    }
    oss << "}";
    oss << " nelements=" << totalElements;
//    uniqueName = oss.str();
    //    StatefulTimer::timeCheck(oss.str().c_str());
  }
  EasyCL *cl = aInfo.wrapper->getCl();
  CLKernel *kernel = 0;
  if( cl->kernelExists(oss.str()) ) {
    kernel = cl->getKernel(oss.str());
//    StatefulTimer::timeCheck("Apply2 1aa");
  } else {
    string uniqueName = oss.str();
//    StatefulTimer::timeCheck("Apply2 1a");
    TemplatedKernel kernelBuilder(cl);
    kernelBuilder.set("dim1", A);
    kernelBuilder.set("dim2", B);
    std::vector<int> dims;
//    StatefulTimer::timeCheck("Apply2 1b");
    if( A >= 0 ) {
      dims.push_back(A);
    }
    if( B != A && B >= 0 ) {
      dims.push_back(B);
    }
    std::string operation = op->operator2();
    kernelBuilder.set("num_tensors", numTensors);
    kernelBuilder.set("num_scalars", numScalars);
    kernelBuilder.set("dims", dims);
    kernelBuilder.set("num_point_tensors", numPointTensors);
    kernelBuilder.set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS);
    kernelBuilder.set("IndexType", TypeParseTraits<IndexType>::name);
    StatefulTimer::timeCheck("Apply2 1c");
    kernelBuilder.set("WarpSize", 32);
    kernelBuilder.set("operation", operation);
    StatefulTimer::timeCheck("Apply2 2a");
    kernelBuilder.set("include_THClReduceApplyUtils", THClReduceApplyUtils_getKernelTemplate());
    StatefulTimer::timeCheck("Apply2 2");
    StatefulTimer::timeCheck("Apply2 3");
    try {
      kernel = kernelBuilder.buildKernel( uniqueName, uniqueName, get_template(), "THClTensor_pointwiseApplyD" );
    } catch( std::runtime_error &e ) {
      std::cout << "Error building kernel in apply2 " << __FILE__ << ":" << easycl::toString( __LINE__ ) << ": " << e.what() << std::endl;
      THError( ( std::string("Error building kernel in apply2 ") + __FILE__ + ":" + easycl::toString( __LINE__ ) + ": " + e.what() ).c_str() );
  //    throw e;
    }
    StatefulTimer::timeCheck("Apply2 compiled");
  }
//  StatefulTimer::timeCheck("Apply2 5a");
  THClKernels k(state, kernel);
//  StatefulTimer::timeCheck("Apply2 5");
  k.out(aInfo);
  k.in(bInfo);
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
//  StatefulTimer::timeCheck("Apply2 6");
  k.run(grid, block);
//  StatefulTimer::timeCheck("Apply2 7");

  if(state->addFinish) cl->finish();
  if(StatefulTimer::enabled) StatefulTimer::timeCheck(("Apply2 END " + oss.str()).c_str());
}

template< typename IndexType >
void kernelLaunch_pointwiseApply3(THClState *state, const int device, dim3 grid, dim3 block, int A, int B, int C, TensorInfo<IndexType> aInfo, TensorInfo<IndexType> bInfo, TensorInfo<IndexType> cInfo, IndexType totalElements, HasOperator3 const*op) {
  StatefulTimer::timeCheck("Apply3 START");
  int numTensors = 3;
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
  StatefulTimer::timeCheck("Apply3 getname");
  ostringstream oss;
  oss << "Apply_3t_" << numScalars << "s_" << numPointTensors << "pt_" << A << "_" << B << "_" << C << "_" << op->operator3();
  StatefulTimer::timeCheck("Apply3 gotname");
  if(false && StatefulTimer::enabled) {
    oss << " dims=" << aInfo.dims;
    oss << "tensor a: ";
    oss << " sizes={";
    for(int d=0; d < aInfo.dims; d++) {
      if(d > 0) {
        oss << ",";
      }
      oss << aInfo.sizes[d];
    }
    oss << "}";
    oss << " strides={";
    for(int d=0; d < aInfo.dims; d++) {
      if(d > 0) {
        oss << ",";
      }
      oss << aInfo.strides[d];
    }
    oss << "}";
    oss << "tensor b: ";
    oss << " sizes={";
    for(int d=0; d < bInfo.dims; d++) {
      if(d > 0) {
        oss << ",";
      }
      oss << bInfo.sizes[d];
    }
    oss << "}";
    oss << " strides={";
    for(int d=0; d < bInfo.dims; d++) {
      if(d > 0) {
        oss << ",";
      }
      oss << bInfo.strides[d];
    }
    oss << "}";
    oss << "tensor c: ";
    oss << " sizes={";
    for(int d=0; d < cInfo.dims; d++) {
      if(d > 0) {
        oss << ",";
      }
      oss << cInfo.sizes[d];
    }
    oss << "}";
    oss << " strides={";
    for(int d=0; d < cInfo.dims; d++) {
      if(d > 0) {
        oss << ",";
      }
      oss << cInfo.strides[d];
    }
    oss << "}";
    oss << " nelements=" << totalElements;
//    uniqueName = oss.str();
  }
  EasyCL *cl = aInfo.wrapper->getCl();
  CLKernel *kernel = 0;
  if(cl->kernelExists(oss.str())) {
    kernel = cl->getKernel(oss.str());
  } else {
    string uniqueName = oss.str();
    TemplatedKernel kernelBuilder(cl);
    kernelBuilder.set("dim1", A);
    kernelBuilder.set("dim2", B);
    kernelBuilder.set("dim3", C);
    std::vector<int> dims;
    if( A >= 0 ) {
      dims.push_back(A);
    }
    if( B != A && B >= 0 ) {
      dims.push_back(B);
    }
    if( C != A && C != B && C >= 0 ) {
      dims.push_back(C);
    }
    std::string operation = op->operator3();
    kernelBuilder.set("num_tensors", numTensors);
    kernelBuilder.set("num_scalars", numScalars);
    kernelBuilder.set("num_point_tensors", numPointTensors);
    kernelBuilder.set("dims", dims);
    kernelBuilder.set("IndexType", TypeParseTraits<IndexType>::name);
    kernelBuilder.set("WarpSize", 32);
    kernelBuilder.set("MAX_CLTORCH_DIMS", MAX_CLTORCH_DIMS);
    kernelBuilder.set("include_THClReduceApplyUtils", THClReduceApplyUtils_getKernelTemplate());
    kernelBuilder.set("operation", operation);
    kernel = kernelBuilder.buildKernel( uniqueName, uniqueName, get_template(), "THClTensor_pointwiseApplyD" );
//    cout << kernelBuilder.getRenderedKernel(get_template()) << endl;
    StatefulTimer::timeCheck("Apply3 compiled");
  }
  StatefulTimer::timeCheck("Apply3 got kernel");

  THClKernels k(state, kernel);
//  StatefulTimer::timeCheck("Apply3 a");

  struct THClScratchSpace* scratchSpace = THClState_getDeviceScratchSpace(state, device, 0);
  initTensorInfoCl(&scratchSpace->info1, aInfo);
  initTensorInfoCl(&scratchSpace->info2, bInfo);
  initTensorInfoCl(&scratchSpace->info3, cInfo);
  scratchSpace->info1Wrap->copyToDevice();
  scratchSpace->info2Wrap->copyToDevice();
  scratchSpace->info3Wrap->copyToDevice();

  k.in(scratchSpace->info1Wrap);
  k.out(aInfo.wrapper);

  k.in(scratchSpace->info2Wrap);
  k.in(bInfo.wrapper);

  k.in(scratchSpace->info3Wrap);
  k.in(cInfo.wrapper);

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
//  StatefulTimer::timeCheck("Apply3 c");
  k.run(grid, block);

  if(state->addFinish) cl->finish();
  if(StatefulTimer::enabled) StatefulTimer::timeCheck(("Apply3 END " + oss.str()).c_str());
}

bool THClTensor_pointwiseApply1(THClState* state,
                                  THClTensor* a,
                                  const HasOperator1 *op,
                                  TensorArgType aType) {
  const int device = a->storage->device;
  long totalElements = THClTensor_nElement(state, a);

  if (THClTensor_nDimension(state, a) > MAX_CLTORCH_DIMS) {
    return false;
  }

  if (THClTensor_nDimension(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock(state, device);

  dim3 grid;
  if (!getApplyGrid(state, device, totalElements, grid)) {
    return false;
  }

  // If tensor args have overlapping indices and are read/write, then
  // we must expand the tensor to a contiguous form first, since
  // otherwise there are conflicting writes. Upon copying back to the
  // non-contiguous form, there will be conflicting writes, but at
  // least with copy, one of the updaters will win atomically. This is
  // a sketchy property of the old system as well (writing into all
  // indices of a tensor with overlapping indices should probably be
  // an error, since it is unclear which one should win), but we will
  // preserve this last-writer-wins (in arbitrary copy order) behavior.
  THClTensor* oldA = NULL;

  if (aType == ReadWrite && THCL_overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = THClTensor_newContiguous(state, a);
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, A)                                   \
   kernelLaunch_pointwiseApply1<TYPE>(state, grid, block, A, aInfo, (TYPE) totalElements, op ); \
  /*THClTensor_pointwiseApply1<Op, TYPE, A>                    \
    <<<grid, block, 0, THClState_getCurrentStream(state)>>>(    \
      aInfo, (TYPE) totalElements, op);*/

#define HANDLE_A_CASE(TYPE, A)                      \
  {                                                 \
    if (aInfo.isContiguous()) {                     \
      HANDLE_CASE(TYPE, -2);                        \
    } else {                                        \
      switch (A) {                                  \
        case 1:                                     \
          HANDLE_CASE(TYPE, 1);                     \
          break;                                    \
        case 2:                                     \
          HANDLE_CASE(TYPE, 2);                     \
          break;                                    \
        case 3:                                     \
          HANDLE_CASE(TYPE, 3);                     \
          break;                                    \
        default:                                    \
          HANDLE_CASE(TYPE, -1);                    \
          break;                                    \
      }                                             \
    }                                               \
  }

  // Can we use 32-bit integer math in the kernel (the linear ID for the copy
  // and the resulting non-linear offset is all computable using 32-bit math?)
  // We also use unsigned index math in the kernel, as signed div/mod has
  // additional overhead.
  if (THCL_canUse32BitIndexMath(state, a)) {
    TensorInfo<unsigned int> aInfo(state, a);

    HANDLE_A_CASE(unsigned int, aInfo.dims);
  } else {
    TensorInfo<unsigned long> aInfo(state, a);

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous()) {
      /*THClTensor_pointwiseApply1<Op, unsigned long, -2>
        <<<grid, block, 0, THClState_getCurrentStream(state)>>>(
          aInfo, (unsigned long) totalElements, op);*/
      THError("Not implemented");
    } else {
      /*THClTensor_pointwiseApply1<Op, unsigned long, -1>
        <<<grid, block, 0, THClState_getCurrentStream(state)>>>(
          aInfo, (unsigned long) totalElements, op);*/
      THError("Not implemented");
    }
  }
#undef HANDLE_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THClTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    THClTensor_copyIgnoringOverlaps(state, oldA, a);
    THClTensor_free(state, a);
    a = oldA;
  }

  return true;
}

bool THClTensor_pointwiseApply2(THClState* state,
                                  THClTensor* a,
                                  THClTensor* b,
                                  const HasOperator2 *op,
                                  TensorArgType aType,
                                  TensorArgType bType) {
  long totalElements = THClTensor_nElement(state, a);
  const int device = b->storage->device;

  if (totalElements != THClTensor_nElement(state, b)) {
    std::cout << "apply2 num elements mismatch" << std::endl;
    return false;
  }

  if (THClTensor_nDimension(state, a) > MAX_CLTORCH_DIMS ||
      THClTensor_nDimension(state, b) > MAX_CLTORCH_DIMS) {
    std::cout << "apply2 too many dimensions" << std::endl;
    return false;
  }

  if (THClTensor_nDimension(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock(state, device);

  dim3 grid;
  if (!getApplyGrid(state, device, totalElements, grid)) {
    std::cout << "apply2 couldnt get apply grid" << std::endl;
    return false;
  }

  // If tensor args have overlapping indices and are read/write, then
  // we must expand the tensor to a contiguous form first, since
  // otherwise there are conflicting writes. Upon copying back to the
  // non-contiguous form, there will be conflicting writes, but at
  // least with copy, one of the updaters will win atomically. This is
  // a sketchy property of the old system as well (writing into all
  // indices of a tensor with overlapping indices should probably be
  // an error, since it is unclear which one should win), but we will
  // preserve this last-writer-wins (in arbitrary copy order) behavior.
  THClTensor* oldA = NULL;
  THClTensor* oldB = NULL;

  if (aType == ReadWrite && THCL_overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = THClTensor_newContiguous(state, a);
  }
  if (bType == ReadWrite && THCL_overlappingIndices(state, b)) {
    // Must perform in contiguous space
    oldB = b;
    b = THClTensor_newContiguous(state, b);
  }

  // It is possible that the tensor dimensions are able to be collapsed,
  // and thus we can reduce the actual code complexity of the copy by
  // exploiting this knowledge statically, since the div/mod is the
  // most expensive part of the operation, more so than memory accesses.
  // For instance, when copying a non-contiguous to a contiguous tensor
  // (or vice versa), the contiguous tensor can be collapsed to one
  // dimension, and the loop to translate the linear index to the array
  // index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, A, B)                                \
   kernelLaunch_pointwiseApply2< TYPE>(state, grid, block, A, B, aInfo, bInfo, (TYPE) totalElements, op ); \
  /* THClTensor_pointwiseApply2<Op, TYPE, A, B>                 \
    <<<grid, block, 0, THClState_getCurrentStream(state)>>>(    \
      aInfo, bInfo, (TYPE) totalElements, op); */

#define HANDLE_B_CASE(TYPE, A, B)                   \
  {                                                 \
    if (bInfo.isContiguous()) {                     \
      HANDLE_CASE(TYPE, A, -2);                     \
    } else {                                        \
      switch (B) {                                  \
        case 1:                                     \
          HANDLE_CASE(TYPE, A, 1);                  \
          break;                                    \
        case 2:                                     \
          HANDLE_CASE(TYPE, A, 2);                  \
          break;                                    \
        case 3:                                     \
          HANDLE_CASE(TYPE, A, 3);                  \
          break;                                    \
        default:                                    \
          HANDLE_CASE(TYPE, A, -1);                 \
          break;                                    \
      }                                             \
    }                                               \
  }

#define HANDLE_A_CASE(TYPE, A, B)                   \
  {                                                 \
    if (aInfo.isContiguous()) {                     \
      HANDLE_B_CASE(TYPE, -2, B);                   \
    } else {                                        \
      switch (A) {                                  \
        case 1:                                     \
          HANDLE_B_CASE(TYPE, 1, B);                \
          break;                                    \
        case 2:                                     \
          HANDLE_B_CASE(TYPE, 2, B);                \
          break;                                    \
        case 3:                                     \
          HANDLE_B_CASE(TYPE, 3, B);                \
          break;                                    \
        default:                                    \
          HANDLE_B_CASE(TYPE, -1, B);               \
          break;                                    \
      }                                             \
    }                                               \
  }

  if (THCL_canUse32BitIndexMath(state, a) &&
      THCL_canUse32BitIndexMath(state, b)) {
    TensorInfo<unsigned int> aInfo(state, a);
    TensorInfo<unsigned int> bInfo(state, b);

    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims);
  } else {
    TensorInfo<unsigned long> aInfo(state, a);
    TensorInfo<unsigned long> bInfo(state, b);

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous() && bInfo.isContiguous()) {
      THError("Not implemented");
//      THClTensor_pointwiseApply2<Op, unsigned long, -2, -2>
//        <<<grid, block, 0, THClState_getCurrentStream(state)>>>(
//          aInfo, bInfo, (unsigned long) totalElements, op);
    } else {
      THError("Not implemented");
//      THClTensor_pointwiseApply2<Op, unsigned long, -1, -1>
//        <<<grid, block, 0, THClState_getCurrentStream(state)>>>(
//          aInfo, bInfo, (unsigned long) totalElements, op);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THClTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    THClTensor_copyIgnoringOverlaps(state, oldA, a);
    THClTensor_free(state, a);
    a = oldA;
  }

  if (oldB) {
    // Ignore overlaps when copying back; if we use THClTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    THClTensor_copyIgnoringOverlaps(state, oldB, b);
    THClTensor_free(state, b);
    b = oldB;
  }

  return true;
}

bool THClTensor_pointwiseApply3(THClState* state,
                                  THClTensor* a,
                                  THClTensor* b,
                                  THClTensor* c,
                                  const HasOperator3 *op,
                                  TensorArgType aType,
                                  TensorArgType bType,
                                  TensorArgType cType) {
  long totalElements = THClTensor_nElement(state, a);
  const int device = b->storage->device;
  if (totalElements != THClTensor_nElement(state, b) ||
      totalElements != THClTensor_nElement(state, c)) {
    std::cout << "element size mismatch between b and c" << std::endl;
    return false;
  }

  if (THClTensor_nDimension(state, a) > MAX_CLTORCH_DIMS ||
      THClTensor_nDimension(state, b) > MAX_CLTORCH_DIMS ||
      THClTensor_nDimension(state, c) > MAX_CLTORCH_DIMS) {
    std::cout << "too many dimensions" << std::endl;
    return false;
  }

  if (THClTensor_nDimension(state, a) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  const dim3 block = getApplyBlock(state, device);

  dim3 grid;
  if (!getApplyGrid(state, device, totalElements, grid)) {
    std::cout << "getapplygrid returns false" << std::endl;
    return false;
  }

  // If tensor args have overlapping indices and are read/write, then
  // we must expand the tensor to a contiguous form first, since
  // otherwise there are conflicting writes. Upon copying back to the
  // non-contiguous form, there will be conflicting writes, but at
  // least with copy, one of the updaters will win atomically. This is
  // a sketchy property of the old system as well (writing into all
  // indices of a tensor with overlapping indices should probably be
  // an error, since it is unclear which one should win), but we will
  // preserve this last-writer-wins (in arbitrary copy order) behavior.
  THClTensor* oldA = NULL;
  THClTensor* oldB = NULL;
  THClTensor* oldC = NULL;

  if (aType == ReadWrite && THCL_overlappingIndices(state, a)) {
    // Must perform in contiguous space
    oldA = a;
    a = THClTensor_newContiguous(state, a);
  }

  if (bType == ReadWrite && THCL_overlappingIndices(state, b)) {
    // Must perform in contiguous space
    oldB = b;
    b = THClTensor_newContiguous(state, b);
  }

  if (cType == ReadWrite && THCL_overlappingIndices(state, c)) {
    // Must perform in contiguous space
    oldC = c;
    c = THClTensor_newContiguous(state, c);
  }

#define HANDLE_CASE(TYPE, A, B, C)                                      \
    /* kernel launch ... */ \
   kernelLaunch_pointwiseApply3<TYPE>(state, device, grid, block, A, B, C, aInfo, bInfo, cInfo, (TYPE) totalElements, op ); \
  /* THClTensor_pointwiseApply3<Op, TYPE, A, B, C> */                      \
    /* <<<grid, block, 0, THClState_getCurrentStream(state)>>>(             \
      aInfo, bInfo, cInfo, (TYPE) totalElements, op); */

#define HANDLE_C_CASE(TYPE, A, B, C)             \
  {                                              \
    if (cInfo.isContiguous()) {                  \
      HANDLE_CASE(TYPE, A, B, -2);               \
    } else {                                     \
      switch (C) {                               \
        case 1:                                  \
          HANDLE_CASE(TYPE, A, B, 1);            \
          break;                                 \
        case 2:                                  \
          HANDLE_CASE(TYPE, A, B, 2);            \
          break;                                 \
        case 3:                                  \
          HANDLE_CASE(TYPE, A, B, 3);            \
          break;                                 \
        default:                                 \
          HANDLE_CASE(TYPE, A, B, -1);           \
          break;                                 \
      }                                          \
    }                                            \
  }

#define HANDLE_B_CASE(TYPE, A, B, C)                 \
  {                                                  \
    if (bInfo.isContiguous()) {                      \
      HANDLE_C_CASE(TYPE, A, -2, C);                 \
    } else {                                         \
      switch (B) {                                   \
        case 1:                                      \
          HANDLE_C_CASE(TYPE, A, 1, C);              \
          break;                                     \
        case 2:                                      \
          HANDLE_C_CASE(TYPE, A, 2, C);              \
          break;                                     \
        case 3:                                      \
          HANDLE_C_CASE(TYPE, A, 3, C);              \
          break;                                     \
        default:                                     \
          HANDLE_C_CASE(TYPE, A, -1, C);             \
          break;                                     \
      }                                              \
    }                                                \
  }

#define HANDLE_A_CASE(TYPE, A, B, C)                 \
  {                                                  \
    if (aInfo.isContiguous()) {                      \
      HANDLE_B_CASE(TYPE, -2, B, C);                 \
    } else {                                         \
      switch (A) {                                   \
        case 1:                                      \
          HANDLE_B_CASE(TYPE, 1, B, C);              \
          break;                                     \
        case 2:                                      \
          HANDLE_B_CASE(TYPE, 2, B, C);              \
          break;                                     \
        case 3:                                      \
          HANDLE_B_CASE(TYPE, 3, B, C);              \
          break;                                     \
        default:                                     \
          HANDLE_B_CASE(TYPE, -1, B, C);             \
          break;                                     \
      }                                              \
    }                                                \
  }

  if (THCL_canUse32BitIndexMath(state, a) &&
      THCL_canUse32BitIndexMath(state, b) &&
      THCL_canUse32BitIndexMath(state, c)) {
    TensorInfo<unsigned int> aInfo(state, a);
    TensorInfo<unsigned int> bInfo(state, b);
    TensorInfo<unsigned int> cInfo(state, c);

    HANDLE_A_CASE(unsigned int, aInfo.dims, bInfo.dims, cInfo.dims);
  } else {
    TensorInfo<unsigned long> aInfo(state, a);
    TensorInfo<unsigned long> bInfo(state, b);
    TensorInfo<unsigned long> cInfo(state, c);

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (aInfo.isContiguous() && bInfo.isContiguous() && cInfo.isContiguous()) {
      THError("Not implemented");
//      THClTensor_pointwiseApply3<Op, unsigned long, -2, -2, -2>
//        <<<grid, block, 0, THClState_getCurrentStream(state)>>>(
//          aInfo, bInfo, cInfo, (unsigned long) totalElements, op);
    } else {
      THError("Not implemented");
//      THClTensor_pointwiseApply3<Op, unsigned long, -1, -1, -1>
//        <<<grid, block, 0, THClState_getCurrentStream(state)>>>(
//          aInfo, bInfo, cInfo, (unsigned long) totalElements, op);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_C_CASE
#undef HANDLE_B_CASE
#undef HANDLE_A_CASE

  if (oldA) {
    // Ignore overlaps when copying back; if we use THClTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldA contiguous.
    THClTensor_copyIgnoringOverlaps(state, oldA, a);
    THClTensor_free(state, a);
    a = oldA;
  }

  if (oldB) {
    // Ignore overlaps when copying back; if we use THClTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldB contiguous.
    THClTensor_copyIgnoringOverlaps(state, oldB, b);
    THClTensor_free(state, b);
    b = oldB;
  }

  if (oldC) {
    // Ignore overlaps when copying back; if we use THClTensor_copy
    // instead, it will recursively try and invoke ourselves to make
    // oldC contiguous.
    THClTensor_copyIgnoringOverlaps(state, oldC, c);
    THClTensor_free(state, c);
    c = oldC;
  }

  return true;
}

#undef THCL_APPLY_THREADS_PER_BLOCK

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
  "// not used by this kernel, but used by THClReduceApplyUtils...\n" 
  "//inline float reduceOp(float _in1, float _in2) {\n" 
  "//  return 0;\n" 
  "//}\n" 
  "\n" 
  "{{include_THClReduceApplyUtils}}\n" 
  "\n" 
  "{%\n" 
  " total_opsize = num_tensors\n" 
  " if include_scalar_input then\n" 
  "      total_opsize = total_opsize + 1\n" 
  "   end\n" 
  " %}\n" 
  "\n" 
  "inline void op( global float *out\n" 
  "  {% for i=1,(num_tensors-1) do %}\n" 
  "  , global float *in{{i}}\n" 
  "  {% end %}\n" 
  "  {% for i=1,(num_scalars) do %}\n" 
  "  , float val{{i}}\n" 
  "  {% end %}\n" 
  "   {% for i=1,num_point_tensors do %}\n" 
  "   , global float *pointTensor{{i}}\n" 
  "   {% end %}\n" 
  ") {\n" 
  "    {{operation}};\n" 
  "}\n" 
  "\n" 
  "kernel void\n" 
  "THClTensor_pointwiseApplyD(\n" 
  "   {% for input_idx=1,num_tensors do %}\n" 
  "    global TensorInfoCl *info_{{input_idx}},\n" 
  "    global float*data_{{input_idx}},\n" 
  "   {% end %}\n" 
  "   {% for i=1,num_scalars do %}\n" 
  "   float val{{i}},\n" 
  "   {% end %}\n" 
  "   {% for i=1,num_point_tensors do %}\n" 
  "   global float *pointTensor{{i}},\n" 
  "   {% end %}\n" 
  "   int totalElements) {\n" 
  "//  for (int linearIndex = get_global_id(0);\n" 
  "   int linearIndex = get_global_id(0);\n" 
  "   if(linearIndex < totalElements ) {\n" 
  "//       linearIndex += get_global_size(0)) {\n" 
  "    {% for input_idx=1,num_tensors do %}\n" 
  "//  int offset{{input_idx}} = linearIndex;\n" 
  "    // Convert `linearIndex` into an offset of `a`\n" 
  "    const int offset{{input_idx}} =\n" 
  "      IndexToOffset_{{1000+loadstring('return dim' .. input_idx)()}}_get(linearIndex, &info_{{input_idx}}[0]);\n" 
  "    {% end %}\n" 
  "\n" 
  "    op(\n" 
  "      {% for input_idx=1,num_tensors do %}\n" 
  "         {% if input_idx > 1 then %} , {% end %}\n" 
  "         &(data_{{input_idx}}[offset{{input_idx}}])\n" 
  "      {% end %}\n" 
  "      {% for i=1,num_scalars do %}\n" 
  "      , val{{i}}\n" 
  "      {% end %}\n" 
  "       {% for i=1,num_point_tensors do %}\n" 
  "       , pointTensor{{i}}\n" 
  "       {% end %}\n" 
  "    );\n" 
  "  }\n" 
  "}\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}


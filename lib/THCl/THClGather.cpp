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

  THClTensor_fill(state, self, 0);

  // since self is write-only, and index and src are read-only, ie none are read-write
  // so, we dnot need to worry about contiguity (at least, not from point of view of correctness)
  
}


//  THArgCheck(indices->nDimension == 1, 3, "expecting vector of indices");
//  THArgCheck(dim < src->nDimension, 4, "Indexing dim is out of bounds");
//  THArgCheck(src->nDimension > 0, 2, "Source tensor is empty");


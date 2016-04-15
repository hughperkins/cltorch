#include <string>
#include <iostream>

#include "THClDebug.h"
#include "THClTensorCopy.h"

using namespace std;

// stuff used during debugging only, not used for prod
// it doesnt clean up after itself or anything, will leak horribly (for now), but
// at least produces useful diagnostic info

void THCl_printTensor(THClState *state, THClTensor *target) {
//    cout << "indices" << endl;
//    cout << THClTensor_toString(state, target) << endl;
    THFloatTensor *probe = THFloatTensor_new();
  THLongStorage *idxsize = THClTensor_newSizeOf(state, target);
THLongStorage *idxstride = THClTensor_newStrideOf(state, target);
    THFloatTensor_resize(probe, idxsize, idxstride);
    THFloatTensor_copyCl(state, probe, target);

    int indices_dim = THFloatTensor_nDimension(probe);
//    cout << "indices_dim " << indices_dim << endl;
    if(indices_dim == 1) {
       cout << " ";
       long size0 = THFloatTensor_size(probe, 0);
       for(int i = 0; i < size0; i++) {
         cout << probe->storage->data[i] << "  ";
      }
//       cout << endl;
      cout << "[torch.ClTensor of size " << size0 << "]" << endl;
    } else if( indices_dim == 2) {
       long size0 = THFloatTensor_size(probe, 0);
       long size1 = THFloatTensor_size(probe, 1);
       for(int i = 0; i < size0; i++) {
         cout << " ";
         for(int j = 0; j < size1; j++) {
            cout << probe->storage->data[i * THFloatTensor_stride(probe, 0) + j * THFloatTensor_stride(probe, 1)] << "  ";
         }
         cout << endl;
      }
      cout << "[torch.ClTensor of size " << size0 << "x" << size1 << "]" << endl;
    } else {
       cout << "target dim > 2" << endl;
    }
}


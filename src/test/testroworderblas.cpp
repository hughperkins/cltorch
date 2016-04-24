// this should be migrated to ues eg googletest framework, but at least it works, for now

#include "THClTensor.h"
#include "THClBlas.h"
#include "THClDebug.h"
#include "THClBlas.h"

#include <iostream>
using namespace std;

void testrows(THClState *state) {
  THClTensor *a = THClTensor_newWithSize2d(state, 0, 2, 3);
  double values[6] = {2, 3, 4, 5, 9, 1};
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 3; j++) {
      THClTensor_set2d(state, a, i, j, (float)values[i * 3 + j]);
    }
  }
  THClDebug_printTensor(state, a);
  double b_values[6] = {-1, 0, 3, 7,4,-2};
  THClTensor *b = THClTensor_newWithSize2d(state, 0, 3, 2);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 2; j++) {
      THClTensor_set2d(state, b, i, j, (float)b_values[i * 2 + j]);
    }
  }
  THClDebug_printTensor(state, b);
  THClTensor *c = THClTensor_newWithSize2d(state, 0, 2, 2);
  
  THClBlas_gemm2(state, 'r', 'n','n', 2, 2, 3,   1,
  a, 3, b, 2, 0, c, 2);
  THClDebug_printTensor(state, c);

  double cvalues[4] = {2*(-1)+3*3+4*4,3*7-8,5*(-1)+9*3+1*4,9*7-2};
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2; j++) {
      cout << cvalues[i * 2 + j] << " ";
    }
    cout << endl;
  }
}

void testcols(THClState *state) {
  THClTensor *a = THClTensor_newWithSize2d(state, 0, 2, 3);
  double avalues[6] = {2, 3, 4, 5, 9, 1};
  for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 3; j++) {
      THClTensor_set2d(state, a, i, j, (float)avalues[i * 3 + j]);
    }
  }
  THClTensor *adash = THClTensor_newTranspose(state, a, 0, 1);
  THClDebug_printTensor(state, adash);
  double bvalues[6] = {-1, 0, 3, 7,4,-2};
  THClTensor *b = THClTensor_newWithSize2d(state, 0, 3, 2);
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 2; j++) {
      THClTensor_set2d(state, b, i, j, (float)bvalues[i * 2 + j]);
    }
  }
  THClTensor *bdash = THClTensor_newTranspose(state, b, 0, 1);
  THClDebug_printTensor(state, bdash);
  THClTensor *c = THClTensor_newWithSize2d(state, 0, 3, 3);
  
  THClBlas_gemm2(state, 'c', 'n','n', 3, 3, 2,   1,
  b, 3, a, 2, 0, c, 3);
  THClTensor *cdash = THClTensor_newTranspose(state, c, 0, 1);
  THClDebug_printTensor(state, cdash);

  double cvalues[9];
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      double sum = 0;
      cout << "   ";
      for(int k = 0 ; k < 2; k++ ) {
//        cout << " " << avalues[k*3+i] << "*" << bvalues[j*2 + k] << ",";
        sum += bvalues[k*3+i] * avalues[j*2 + k];
      }
//      cout << endl;
//      cout << "i=" << i << " j=" << j << " sum=" << sum << endl;
      cvalues[j*3+i] = sum;
    }
  }
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
//      cvalues[j*3+i] = avalues[j*2+i] * 
      cout << cvalues[i * 3 + j] << " ";
    }
    cout << endl;
  }
}

int main(int argc, char *argv[]) {
  THClState* state = (THClState*)malloc(sizeof(THClState));
  THClInit(state);

  testrows(state);
  testcols(state);

  return 0;
}


THCL_API THClTensor *THClTensor_newWithSize2d(THClState *state, int device, long size0_, long size1_);


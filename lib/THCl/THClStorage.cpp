#include "THClStorage.h"
#include "THClGeneral.h"
#include "THAtomic.h"

#include "EasyCL.h"
#include <stdexcept>
#include <iostream>
using namespace std;

//int state->trace = 0;

//// note to self: this function implementation is a bit rubbish...
//void THClStorage_set(THClState *state, THClStorage *self, long index, float value)
//{
////  cout << "set size=" << self->size << " index=" << index << " value=" << value << endl;
//  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
//  if( self->wrapper->isDeviceDirty() ) { // we have to do this, since we're going to copy it all back again
//                                         // although I suppose we could set via a kernel perhaps
//                                         // either way, this function is pretty inefficient right now :-P
//    if(state->trace) cout << "wrapper->copyToHost() size " << self->size << endl;
//    self->wrapper->copyToHost();
//  }
//  self->data[index] = value;
//  if(state->trace) cout << "wrapper->copyToDevice() size " << self->size << endl;
//  self->wrapper->copyToDevice();
//}

//// note to self: this function implementation is a bit rubbish...
//float THClStorage_get(THClState *state, const THClStorage *self, long index)
//{
////  printf("THClStorage_get\n");
//  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
//  if( self->wrapper->isDeviceDirty() ) {
//    if(state->trace) cout << "wrapper->copyToHost()" << endl;
//    self->wrapper->copyToHost();
//  }
//  return self->data[index];
//}

THClStorage* THClStorage_new(THClState *state)
{
  return THClStorage_newv2(state, state->currentDevice);
}

THClStorage* THClStorage_newv2(THClState *state, const int device)
{
  THClStorage *storage = (THClStorage*)THAlloc(sizeof(THClStorage));
  storage->device = device;
  storage->cl = THClState_getClv2(state, storage->device);
//  storage->device = -1;
  storage->data = NULL;
  storage->wrapper = 0;
  storage->size = 0;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  return storage;
}

THClStorage* THClStorage_newWithSize(THClState *state, int device, long size)
{
  THArgCheck(size >= 0, 2, "invalid size");

  if(size > 0)
  {
    THClStorage *storage = (THClStorage*)THAlloc(sizeof(THClStorage));
    float *data = new float[size];
    storage->device = device;
    storage->cl = THClState_getClv2(state, storage->device);
    CLWrapper *wrapper = storage->cl->wrap( size, data );
    if(state->trace) cout << "new wrapper, size " << size << endl;
    if(state->trace) cout << "wrapper->createOnDevice()" << endl;
    wrapper->createOnDevice();
    storage->data = data;
    storage->wrapper = wrapper;

    storage->size = size;
    storage->refcount = 1;
    storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
    return storage;
  }
  else
  {
    return THClStorage_newv2(state, device);
  }
}

THClStorage* THClStorage_newWithSize1(THClState *state, int device, float data0)
{
  THError("not available yet for THClStorage");
  return NULL;
//  THClStorage *self = THClStorage_newWithSize(state, 1);
//  THClStorage_set(state, self, 0, data0);
//  return self;
}

THClStorage* THClStorage_newWithSize2(THClState *state, int device, float data0, float data1)
{
  THError("not available yet for THClStorage");
  return NULL;
//  THClStorage *self = THClStorage_newWithSize(state, 2);
//  THClStorage_set(state, self, 0, data0);
//  THClStorage_set(state, self, 1, data1);
//  return self;
}

THClStorage* THClStorage_newWithSize3(THClState *state, int device, float data0, float data1, float data2)
{
  THError("not available yet for THClStorage");
  return NULL;
//  THClStorage *self = THClStorage_newWithSize(state, 3);
//  THClStorage_set(state, self, 0, data0);
//  THClStorage_set(state, self, 1, data1);
//  THClStorage_set(state, self, 2, data2);
//  return self;
}

THClStorage* THClStorage_newWithSize4(THClState *state, int device, float data0, float data1, float data2, float data3)
{
  THError("not available yet for THClStorage");
  return NULL;
//  THClStorage *self = THClStorage_newWithSize(state, 4);
//  THClStorage_set(state, self, 0, data0);
//  THClStorage_set(state, self, 1, data1);
//  THClStorage_set(state, self, 2, data2);
//  THClStorage_set(state, self, 3, data3);
//  return self;
}

THClStorage* THClStorage_newWithMapping(THClState *state, int device, const char *fileName, long size, int isShared)
{
  THError("not available yet for THClStorage");
  return NULL;
}

THClStorage* THClStorage_newWithData(THClState *state, int device, float *data, long size)
{
  THError("not available yet for THClStorage");
  return NULL;
//  THClStorage *storage = (THClStorage*)THAlloc(sizeof(THClStorage));
//  storage->data = data;
//  storage->size = size;
//  storage->refcount = 1;
//  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
//  return storage;
}

void THClStorage_retain(THClState *state, THClStorage *self)
{
  if(self && (self->flag & TH_STORAGE_REFCOUNTED))
    THAtomicIncrementRef(&self->refcount);
}

void THClStorage_free(THClState *state, THClStorage *self)
{
  if(!(self->flag & TH_STORAGE_REFCOUNTED))
    return;

  if (THAtomicDecrementRef(&self->refcount))
  {
    if(self->flag & TH_STORAGE_FREEMEM) {
      if(state->trace && self->size > 0) cout << "delete wrapper, size " << self->size << endl;
      delete self->wrapper;
      delete self->data;
    }
    THFree(self);
  }
}
void THClStorage_fill(THClState *state, THClStorage *self, float value)
{
  for( int i = 0; i < self->size; i++ ) {
    self->data[i] = value;
  }
  self->wrapper->copyToDevice();
  if(state->trace) cout << "wrapper->copyToDevice() size" << self->size << endl;
}

void THClStorage_resize(THClState *state, THClStorage *self, long size)
{
  if( size <= self->size ) {
    return;
  }
  delete self->wrapper;
  if(state->trace && self->size > 0) cout << "delete wrapper" << endl;
  delete[] self->data;
  self->data = new float[size];
  EasyCL *cl = self->cl;
  self->wrapper = cl->wrap( size, self->data );
  self->wrapper->createOnDevice();
    if(state->trace) cout << "new wrapper, size " << size << endl;
  self->size = size;
}


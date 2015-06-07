#include "THClStorage.h"
#include "THClGeneral.h"
#include "THAtomic.h"

#include "EasyCL.h"
#include <stdexcept>
//#include <iostream>
using namespace std;

void THClStorage_set(THClState *state, THClStorage *self, long index, float value)
{
//  cout << "set size=" << self->size << " index=" << index << " value=" << value << endl;
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  if( self->wrapper->isDeviceDirty() ) { // we have to do this, since we're going to copy it all back again
                                         // although I suppose we could set via a kernel perhaps
                                         // either way, this function is pretty inefficient right now :-P
    self->wrapper->copyToHost();
  }
  self->data[index] = value;
  self->wrapper->copyToDevice();
}

float THClStorage_get(THClState *state, const THClStorage *self, long index)
{
//  printf("THClStorage_get\n");
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  if( self->wrapper->isDeviceDirty() ) {
    self->wrapper->copyToHost();
  }
  return self->data[index];
}

THClStorage* THClStorage_new(THClState *state)
{
  THClStorage *storage = (THClStorage*)THAlloc(sizeof(THClStorage));
  storage->data = NULL;
  storage->wrapper = 0;
  storage->size = 0;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  return storage;
}

THClStorage* THClStorage_newWithSize(THClState *state, long size)
{
  THArgCheck(size >= 0, 2, "invalid size");

  if(size > 0)
  {
    THClStorage *storage = (THClStorage*)THAlloc(sizeof(THClStorage));
    float *data = new float[size];
    EasyCL *cl = state->cl;
    CLWrapper *wrapper = cl->wrap( size, data );
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
    return THClStorage_new(state);
  }
}

THClStorage* THClStorage_newWithSize1(THClState *state, float data0)
{
  THError("not available yet for THClStorage");
  return NULL;
//  THClStorage *self = THClStorage_newWithSize(state, 1);
//  THClStorage_set(state, self, 0, data0);
//  return self;
}

THClStorage* THClStorage_newWithSize2(THClState *state, float data0, float data1)
{
  THError("not available yet for THClStorage");
  return NULL;
//  THClStorage *self = THClStorage_newWithSize(state, 2);
//  THClStorage_set(state, self, 0, data0);
//  THClStorage_set(state, self, 1, data1);
//  return self;
}

THClStorage* THClStorage_newWithSize3(THClState *state, float data0, float data1, float data2)
{
  THError("not available yet for THClStorage");
  return NULL;
//  THClStorage *self = THClStorage_newWithSize(state, 3);
//  THClStorage_set(state, self, 0, data0);
//  THClStorage_set(state, self, 1, data1);
//  THClStorage_set(state, self, 2, data2);
//  return self;
}

THClStorage* THClStorage_newWithSize4(THClState *state, float data0, float data1, float data2, float data3)
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

THClStorage* THClStorage_newWithMapping(THClState *state, const char *fileName, long size, int isShared)
{
  THError("not available yet for THClStorage");
  return NULL;
}

THClStorage* THClStorage_newWithData(THClState *state, float *data, long size)
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
}

void THClStorage_resize(THClState *state, THClStorage *self, long size)
{
  if( size <= self->size ) {
    return;
  }
  delete self->wrapper;
  delete[] self->data;
  self->data = new float[size];
  self->wrapper = state->cl->wrap( size, self->data );
  self->size = size;
}


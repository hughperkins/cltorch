#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tensor.c"
#else

#include <stdio.h>
#include <iostream>
using namespace std;

static THTensor *getTensor(lua_State *L, int index) {
  void *tensorVoid = luaT_checkudata(L, index, torch_Tensor);
  return (THTensor *)tensorVoid;
}

//static THClTensor *getClTensor(lua_State *L, int index ) {
//  void *clTensorVoid = luaT_checkudata(L, index, "torch.ClTensor");
//  return (THClTensor *)clTensorVoid;
//}

//static THByteTensor *getByteTensor(lua_State *L, int index ) {
//  void *byteTensorVoid = luaT_checkudata(L, index, "torch.ByteTensor");
//  return (THByteTensor *)byteTensorVoid;
//}

//static THLongTensor *getLongTensor(lua_State *L, int index ) {
//  void *longTensorVoid = luaT_checkudata(L, index, "torch.LongTensor");
//  return (THLongTensor *)longTensorVoid;
//}


static THTensor *getTensorNoCheck(lua_State *L, int index) {
  void *tensorVoid = luaT_toudata(L, index, torch_Tensor);
  return (THTensor *)tensorVoid;
}

static THClTensor *getClTensorNoCheck(lua_State *L, int index ) {
  void *clTensorVoid = luaT_toudata(L, index, "torch.ClTensor");
  return (THClTensor *)clTensorVoid;
}

static THByteTensor *getByteTensorNoCheck(lua_State *L, int index ) {
  void *byteTensorVoid = luaT_toudata(L, index, "torch.ByteTensor");
  return (THByteTensor *)byteTensorVoid;
}

static THLongTensor *getLongTensorNoCheck(lua_State *L, int index ) {
  void *longTensorVoid = luaT_toudata(L, index, "torch.LongTensor");
  return (THLongTensor *)longTensorVoid;
}


//static THStorage *getStorage(lua_State *L, int index) {
//  void *storageVoid = luaT_checkudata(L, index, torch_Storage);
//  return (THStorage *)storageVoid;
//}

//static THLongStorage *getLongStorage(lua_State *L, int index) {
//  void *longStorageVoid = luaT_checkudata(L, index, "torch.LongStorage");
//  return (THLongStorage *)longStorageVoid;
//}


static THStorage *getStorageNoCheck(lua_State *L, int index) {
  void *storageVoid = luaT_toudata(L, index, torch_Storage);
  return (THStorage *)storageVoid;
}

static THLongStorage *getLongStorageNoCheck(lua_State *L, int index) {
  void *longStorageVoid = luaT_toudata(L, index, "torch.LongStorage");
  return (THLongStorage *)longStorageVoid;
}

static void torch_Tensor_(c_readTensorStorageSizeStride)(lua_State *L, int index, int allowNone, int allowTensor, int allowStorage, int allowStride,
                                                         THStorage **storage_, long *storageOffset_, THLongStorage **size_, THLongStorage **stride_);

static void torch_Tensor_(c_readSizeStride)(lua_State *L, int index, int allowStride, THLongStorage **size_, THLongStorage **stride_);

static int torch_Tensor_(size)(lua_State *L)
{
  THTensor *tensor = getTensor(L, 1);
  if(lua_isnumber(L,2))
  {
    int dim = luaL_checkint(L, 2)-1;
    luaL_argcheck(L, dim >= 0 && dim < tensor->nDimension, 2, "out of range");
    lua_pushnumber(L, tensor->size[dim]);
  }
  else
  {
    THLongStorage *storage = THLongStorage_newWithSize(tensor->nDimension);
    memmove(storage->data, tensor->size, sizeof(long)*tensor->nDimension);
    luaT_pushudata(L, storage, "torch.LongStorage");
  }
  return 1;
}

static int torch_Tensor_(stride)(lua_State *L)
{
  THTensor *tensor = getTensor(L, 1);
  if(lua_isnumber(L,2))
  {
    int dim = luaL_checkint(L, 2)-1;
    luaL_argcheck(L, dim >= 0 && dim < tensor->nDimension, 2, "out of range");
    lua_pushnumber(L, tensor->stride[dim]);
  }
  else
  {
    THLongStorage *storage = THLongStorage_newWithSize(tensor->nDimension);
    memmove(storage->data, tensor->stride, sizeof(long)*tensor->nDimension);
    luaT_pushudata(L, storage, "torch.LongStorage");
  }
  return 1;
}

static int torch_Tensor_(nDimension)(lua_State *L)
{
  THTensor *tensor = getTensor(L, 1);
  lua_pushnumber(L, tensor->nDimension);
  return 1;
}

static int torch_Tensor_(storage)(lua_State *L)
{
  THTensor *tensor = getTensor(L, 1);
  if(tensor->storage)
  {
    THStorage_(retain)(cltorch_getstate(L), tensor->storage);
    luaT_pushudata(L, tensor->storage, torch_Storage);
  }
  else
    lua_pushnil(L);

  return 1;
}

static int torch_Tensor_(storageOffset)(lua_State *L)
{
  THTensor *tensor = getTensor(L, 1);
  lua_pushnumber(L, tensor->storageOffset+1);
  return 1;
}

static int torch_Tensor_(new)(lua_State *L)
{
  try {
    //  printf("cltorch/torch/generic/Tensor.c torch_Tensor_(new)\n");
    THClState *state = cltorch_getstate(L);
    const int device = state->currentDevice;
    THTensor *tensor;
    long storageOffset;
    THLongStorage *size, *stride;

    if(lua_type(L, 1) == LUA_TTABLE)
    {
      long i, j;
      THLongStorage *counter;
      long si = 0;
      int dimension = 0;
      int is_finished = 0;

      lua_settop(L, 1);
      size = THLongStorage_new();

      while( (lua_type(L, -1) == LUA_TTABLE) && (lua_objlen(L, -1) > 0) )
      {
        THLongStorage_resize(size, dimension+1);
        size->data[dimension] = lua_objlen(L, -1);
        dimension++;
        lua_rawgeti(L, -1, 1);
      }
      lua_pop(L, 1);

      counter = THLongStorage_newWithSize(size->size);
      THLongStorage_fill(counter, 0);

  //    tensor = THTensor_(newWithSize)(state, size, NULL);
      THFloatTensor *tensor = THFloatTensor_newWithSize(size, NULL);

      if(size->size == 0)
        is_finished = 1;

      while(!is_finished)
      {
        if(!lua_istable(L, -1))
        {
          THLongStorage_free(size);
          THLongStorage_free(counter);
          THFloatTensor_free(tensor);
          luaL_error(L, "invalid tensor definition");
        }

        if((long)lua_objlen(L, -1) != size->data[size->size-1])
        {
          THLongStorage_free(size);
          THLongStorage_free(counter);
          THFloatTensor_free(tensor);
          luaL_error(L, "invalid tensor sizes");
        }

        for(i = 0; i < size->data[size->size-1]; i++)
        {
          lua_rawgeti(L, -1, i+1);
          if(!lua_isnumber(L, -1))
          {
            THLongStorage_free(size);
            THLongStorage_free(counter);
            THFloatTensor_free(tensor);
            luaL_error(L, "invalid element (not a number)");
          }
          THFloatStorage_set(THFloatTensor_storage(tensor), si++, (real)lua_tonumber(L, -1));
          lua_pop(L, 1);
        }

        if(size->size == 1)
          break;

        for(i = size->size-2; i >= 0; i--)
        {
          if(++counter->data[i] == size->data[i])
          {
            if(i == 0)
            {
              is_finished = 1;
              break;
            }
            else
            {
              counter->data[i] = 0;
              lua_pop(L, 1);
            }
          }
          else
          {
            lua_pop(L, 1);
            for(j = i; j < size->size-1; j++)
            {
              if(!lua_istable(L, -1))
              {
                THLongStorage_free(size);
                THLongStorage_free(counter);
                THFloatTensor_free(tensor);
                luaL_error(L, "invalid tensor definition");
              }
              if((long)lua_objlen(L, -1) != size->data[j])
              {
                THLongStorage_free(size);
                THLongStorage_free(counter);
                THFloatTensor_free(tensor);
                luaL_error(L, "invalid tensor sizes");
              }
              lua_rawgeti(L, -1, counter->data[j]+1);
            }
            break;
          }
        }
      }
      
      THTensor *tensorcl = THTensor_(newWithSize)(state, device, size, NULL);
    THTensor_(copyFloat)(state, tensorcl, tensor);
      THFloatTensor_free(tensor);

      THLongStorage_free(size);
      THLongStorage_free(counter);
    luaT_pushudata(L, tensorcl, "torch.ClTensor");
    return 1;

  //    THLongStorage_free(size);
  //    THLongStorage_free(counter);
  //  luaT_pushudata(L, tensor, "torch.FloatTensor");
  //  return 1;


      
  //    THError("Please create like: torch.Tensor(mytable):cl()");

  //    long i, j;
  //    THLongStorage *counter;
  //    long si = 0;
  //    int dimension = 0;
  //    int is_finished = 0;

  //    lua_settop(L, 1);
  //    size = THLongStorage_new();

  //    while( (lua_type(L, -1) == LUA_TTABLE) && (lua_objlen(L, -1) > 0) )
  //    {
  //      THLongStorage_resize(size, dimension+1);
  //      size->data[dimension] = lua_objlen(L, -1);
  //      dimension++;
  //      lua_rawgeti(L, -1, 1);
  //    }
  //    lua_pop(L, 1);

  //    counter = THLongStorage_newWithSize(size->size);
  //    THLongStorage_fill(counter, 0);

  //    tensor = THTensor_(newWithSize)(state, size, NULL);

  //    if(size->size == 0)
  //      is_finished = 1;

  //    while(!is_finished)
  //    {
  //      if(!lua_istable(L, -1))
  //      {
  //        THLongStorage_free(size);
  //        THLongStorage_free(counter);
  //        THTensor_(free)(state, tensor);
  //        luaL_error(L, "invalid tensor definition");
  //      }

  //      if(lua_objlen(L, -1) != size->data[size->size-1])
  //      {
  //        THLongStorage_free(size);
  //        THLongStorage_free(counter);
  //        THTensor_(free)(state, tensor);
  //        luaL_error(L, "invalid tensor sizes");
  //      }

  //      for(i = 0; i < size->data[size->size-1]; i++)
  //      {
  //        lua_rawgeti(L, -1, i+1);
  //        if(!lua_isnumber(L, -1))
  //        {
  //          THLongStorage_free(size);
  //          THLongStorage_free(counter);
  //          THTensor_(free)(state, tensor);
  //          luaL_error(L, "invalid element (not a number)");
  //        }
  //        THStorage_(set)(state, THTensor_(storage)(state, tensor), si++, (real)lua_tonumber(L, -1));
  //        lua_pop(L, 1);
  //      }

  //      if(size->size == 1)
  //        break;

  //      for(i = size->size-2; i >= 0; i--)
  //      {
  //        if(++counter->data[i] == size->data[i])
  //        {
  //          if(i == 0)
  //          {
  //            is_finished = 1;
  //            break;
  //          }
  //          else
  //          {
  //            counter->data[i] = 0;
  //            lua_pop(L, 1);
  //          }
  //        }
  //        else
  //        {
  //          lua_pop(L, 1);
  //          for(j = i; j < size->size-1; j++)
  //          {
  //            if(!lua_istable(L, -1))
  //            {
  //              THLongStorage_free(size);
  //              THLongStorage_free(counter);
  //              THTensor_(free)(state, tensor);
  //              luaL_error(L, "invalid tensor definition");
  //            }
  //            if(lua_objlen(L, -1) != size->data[j])
  //            {
  //              THLongStorage_free(size);
  //              THLongStorage_free(counter);
  //              THTensor_(free)(state, tensor);
  //              luaL_error(L, "invalid tensor sizes");
  //            }
  //            lua_rawgeti(L, -1, counter->data[j]+1);
  //          }
  //          break;
  //        }
  //      }
  //    }

  //    THLongStorage_free(size);
  //    THLongStorage_free(counter);
    }
    else
    {
      THStorage *storage;

      torch_Tensor_(c_readTensorStorageSizeStride)(L, 1, 1, 1, 1, 1,
                                                   &storage, &storageOffset, &size, &stride);

      int device = state->currentDevice;
      if( storage != 0 ) {
        device = storage->device;
      }
      tensor = THTensor_(newWithStorage)(state, device, storage, storageOffset, size, stride);

      THLongStorage_free(size);
      THLongStorage_free(stride);
    }

    luaT_pushudata(L, tensor, torch_Tensor);
    return 1;
  } catch(exception &e) {
    THError("Something went wrong: %s", e.what());
    return 0;
  }
}

static int torch_Tensor_(set)(lua_State *L)
{
  THTensor *self = getTensor(L, 1);
  THStorage *storage;
  long storageOffset;
  THLongStorage *size, *stride;

  torch_Tensor_(c_readTensorStorageSizeStride)(L, 2, 1, 1, 1, 1,
                                               &storage, &storageOffset, &size, &stride);

  THTensor_(setStorage)(cltorch_getstate(L), self, storage, storageOffset, size, stride);

  THLongStorage_free(size);
  THLongStorage_free(stride);

  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_(clone)(lua_State *L)
{
  THTensor *self = getTensor(L, 1);
  EXCEPT_TO_THERROR(self = THTensor_(newClone)(cltorch_getstate(L), self));
  luaT_pushudata(L, self, torch_Tensor);
  return 1;
}

static int torch_Tensor_(contiguous)(lua_State *L)
{
  THTensor *self = getTensor(L, 1);
  EXCEPT_TO_THERROR(self = THTensor_(newContiguous)(cltorch_getstate(L), self));
  luaT_pushudata(L, self, torch_Tensor);
  return 1;
}

/* Resize */
static int torch_Tensor_(resizeAs)(lua_State *L)
{
  THTensor *tensor = getTensor(L, 1);
  THTensor *src = getTensor(L, 2);
  THTensor_(resizeAs)(cltorch_getstate(L), tensor, src);
  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_(resize)(lua_State *L)
{
  THTensor *tensor = getTensor(L, 1);
  THLongStorage *size, *stride;

  torch_Tensor_(c_readSizeStride)(L, 2, 0, &size, &stride);

  EXCEPT_TO_THERROR(THTensor_(resize)(cltorch_getstate(L), tensor, size, stride));

  THLongStorage_free(size);
  THLongStorage_free(stride);

  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_(narrow)(lua_State *L)
{
  THClState *state = cltorch_getstate(L);
  THTensor *tensor = getTensor(L, 1);
  int dimension = luaL_checkint(L, 2)-1;
  long firstIndex = luaL_checklong(L, 3)-1;
  long size = luaL_checklong(L, 4);

/*  THArgCheck( (dimension >= 0) && (dimension < tensor->nDimension), 2, "out of range");
  THArgCheck( (firstIndex >= 0) && (firstIndex < tensor->size[dimension]), 3, "out of range");
  THArgCheck( (size > 0) && (firstIndex+size <= tensor->size[dimension]), 4, "out of range");
*/
  EXCEPT_TO_THERROR(tensor = THTensor_(newWithTensor)(state, tensor));
  THTensor_(narrow)(state, tensor, NULL, dimension, firstIndex, size);
  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

static int torch_Tensor_(sub)(lua_State *L)
{
  THClState *state = cltorch_getstate(L);
  THTensor *tensor = getTensor(L, 1);
  long d0s = -1, d0e = -1, d1s = -1, d1e = -1, d2s = -1, d2e = -1, d3s = -1, d3e = -1;

  d0s = luaL_checklong(L, 2)-1;
  d0e = luaL_checklong(L, 3)-1;
  if(d0s < 0)
    d0s += tensor->size[0]+1;
  if(d0e < 0)
    d0e += tensor->size[0]+1;
  luaL_argcheck(L, tensor->nDimension > 0, 2, "invalid dimension");
  luaL_argcheck(L, d0s >= 0 && d0s < tensor->size[0], 2, "out of range");
  luaL_argcheck(L, d0e >= 0 && d0e < tensor->size[0], 3, "out of range");
  luaL_argcheck(L, d0e >= d0s, 3, "end smaller than beginning");

  if(!lua_isnone(L, 4))
  {
    d1s = luaL_checklong(L, 4)-1;
    d1e = luaL_checklong(L, 5)-1;
    if(d1s < 0)
      d1s += tensor->size[1]+1;
    if(d1e < 0)
      d1e += tensor->size[1]+1;
    luaL_argcheck(L, tensor->nDimension > 1, 4, "invalid dimension");
    luaL_argcheck(L, d1s >= 0 && d1s < tensor->size[1], 4, "out of range");
    luaL_argcheck(L, d1e >= 0 && d1e < tensor->size[1], 5, "out of range");
    luaL_argcheck(L, d1e >= d1s, 5, "end smaller than beginning");

    if(!lua_isnone(L, 6))
    {
      d2s = luaL_checklong(L, 6)-1;
      d2e = luaL_checklong(L, 7)-1;
      if(d2s < 0)
        d2s += tensor->size[2]+1;
      if(d2e < 0)
        d2e += tensor->size[2]+1;
      luaL_argcheck(L, tensor->nDimension > 2, 6, "invalid dimension");
      luaL_argcheck(L, d2s >= 0 && d2s < tensor->size[2], 6, "out of range");
      luaL_argcheck(L, d2e >= 0 && d2e < tensor->size[2], 7, "out of range");
      luaL_argcheck(L, d2e >= d2s, 7, "end smaller than beginning");

      if(!lua_isnone(L, 8))
      {
        d3s = luaL_checklong(L, 8)-1;
        d3e = luaL_checklong(L, 9)-1;
        if(d3s < 0)
          d3s += tensor->size[3]+1;
        if(d3e < 0)
          d3e += tensor->size[3]+1;
        luaL_argcheck(L, tensor->nDimension > 3, 8, "invalid dimension");
        luaL_argcheck(L, d3s >= 0 && d3s < tensor->size[3], 8, "out of range");
        luaL_argcheck(L, d3e >= 0 && d3e < tensor->size[3], 9, "out of range");
        luaL_argcheck(L, d3e >= d3s, 9, "end smaller than beginning");
      }
    }
  }

  EXCEPT_TO_THERROR(tensor = THTensor_(newWithTensor)(state, tensor));
  THTensor_(narrow)(state, tensor, NULL, 0, d0s, d0e-d0s+1);
  if(d1s >= 0)
    THTensor_(narrow)(state, tensor, NULL, 1, d1s, d1e-d1s+1);
  if(d2s >= 0)
    THTensor_(narrow)(state, tensor, NULL, 2, d2s, d2e-d2s+1);
  if(d3s >= 0)
    THTensor_(narrow)(state, tensor, NULL, 3, d3s, d3e-d3s+1);
  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

static int torch_Tensor_(select)(lua_State *L)
{
  THClState *state = cltorch_getstate(L);
  THTensor *tensor = getTensor(L, 1);
  int dimension = luaL_checkint(L, 2)-1;
  long sliceIndex = luaL_checklong(L, 3)-1;

/*   THArgCheck(src->nDimension > 1, 1, "cannot select on a vector");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 2, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size[dimension]), 3, "out of range");
*/

  if(tensor->nDimension > 1)
  {
    EXCEPT_TO_THERROR(tensor = THTensor_(newWithTensor)(state, tensor));
    THTensor_(select)(state, tensor, NULL, dimension, sliceIndex);
    luaT_pushudata(L, tensor, torch_Tensor);
  }
  else
  {
    THArgCheck(tensor->nDimension == 1, 1, "empty Tensor");
    lua_pushnumber(L, THTensor_(get1d)(state, tensor, sliceIndex));
  }

  return 1;
}

static int torch_Tensor_(indexSelect)(lua_State *L)
{
  THClState *state = cltorch_getstate(L);
  int narg = lua_gettop(L);
  THTensor *tensor, *src, *index;
  THLongTensor *longIndex;
  int dim;
  if (narg == 3)
  {
    tensor = 0;
    EXCEPT_TO_THERROR(tensor = THTensor_(newv2)(state, state->currentDevice));
    src = getTensor(L, 1);
    dim = luaL_checkint(L, 2) - 1;
    index = getTensorNoCheck(L, 3);
    longIndex = getLongTensorNoCheck(L, 3);
    if (!index && !longIndex) luaT_typerror(L, 3, "LongTensor | Tensor");
    luaT_pushudata(L,tensor,torch_Tensor);
  }
  else if(narg == 4)
  {
    src = getTensor(L, 2);
    dim = luaL_checkint(L, 3) - 1;
    index = getTensorNoCheck(L, 4);
    longIndex = getLongTensorNoCheck(L, 4);
    if (!index && !longIndex) luaT_typerror(L, 4, "Tensor | LongTensor");
    tensor = getTensor(L, 1);
  }
  else
  {
    luaL_error(L, "[Tensor,] Tensor, number, Tensor | LongTensor expected");
    return 0;
  }

  if (index)
    THTensor_(indexSelect)(state, tensor,src,dim,index);
  else
    THTensor_(indexSelect_long)(state, tensor,src,dim,longIndex);

  return 1;
}

static int torch_Tensor_(indexCopy)(lua_State *L)
{
  int narg = lua_gettop(L);
  THTensor *tensor, *src, *index;
  THLongTensor *longIndex;
  int dim;
  if(narg == 4)
  {
    dim = luaL_checkint(L, 2) - 1;
    index = getTensorNoCheck(L, 3);
    longIndex = getLongTensorNoCheck(L, 3);
    if (!index && !longIndex) luaT_typerror(L, 3, "Tensor | LongTensor");
    src = getTensor(L, 4);
    tensor = getTensor(L, 1);
  }
  else
  {
    luaL_error(L,"Tensor, number, Tensor | LongTensor, Tensor expected");
    return 0;
  }

  if (index)
    THTensor_(indexCopy)(cltorch_getstate(L), tensor,dim,index,src);
  else
    THTensor_(indexCopy_long)(cltorch_getstate(L), tensor,dim,longIndex,src);

  return 1;
}

static int torch_Tensor_(indexFill)(lua_State *L)
{
  int narg = lua_gettop(L);
  THTensor *tensor, *index;
  THLongTensor *longIndex;
  real val;
  int dim;
  if(narg == 4)
  {
    dim = luaL_checkint(L, 2) - 1;
    index = getTensorNoCheck(L, 3);
    longIndex = getLongTensorNoCheck(L, 3);
    if (!index && !longIndex) luaT_typerror(L, 3, "Tensor | LongTensor");
    val = luaL_checknumber(L, 4);
    tensor = getTensor(L, 1);
  }
  else
  {
    luaL_error(L,"Tensor, number, Tensor | LongTensor, number expected");
    return 0;
  }

  if (index)
    THTensor_(indexFill)(cltorch_getstate(L), tensor,dim,index,val);
  else
    THTensor_(indexFill_long)(cltorch_getstate(L), tensor,dim,longIndex,val);

  return 1;
}

static int torch_Tensor_(transpose)(lua_State *L)
{
  THClState *state = cltorch_getstate(L);
  THTensor *tensor = getTensor(L, 1);
  int dimension1 = luaL_checkint(L, 2)-1;
  int dimension2 = luaL_checkint(L, 3)-1;

/*
  THArgCheck( (dimension1 >= 0) && (dimension1 < src->nDimension), 2, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < src->nDimension), 3, "out of range");
*/

  EXCEPT_TO_THERROR(tensor = THTensor_(newWithTensor)(state, tensor));
  THTensor_(transpose)(state, tensor, NULL, dimension1, dimension2);
  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

static int torch_Tensor_(t)(lua_State *L)
{
  THClState *state = cltorch_getstate(L);
  THTensor *tensor = getTensor(L, 1);

  luaL_argcheck(L, tensor->nDimension == 2, 1, "Tensor must have 2 dimensions");

  EXCEPT_TO_THERROR(tensor = THTensor_(newWithTensor)(state, tensor));
  THTensor_(transpose)(state, tensor, NULL, 0, 1);
  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

static int torch_Tensor_(unfold)(lua_State *L)
{
  THClState *state = cltorch_getstate(L);
  THTensor *tensor = getTensor(L, 1);
  int dimension = luaL_checkint(L, 2)-1;
  long size = luaL_checklong(L, 3);
  long step = luaL_checklong(L, 4);

/*
  THArgCheck( (src->nDimension > 0), 1, "cannot unfold an empty tensor");
  THArgCheck(dimension < src->nDimension, 2, "out of range");
  THArgCheck(size <= src->size[dimension], 3, "out of range");
*/

  EXCEPT_TO_THERROR(tensor = THTensor_(newWithTensor)(state, tensor));
  THTensor_(unfold)(state, tensor, NULL, dimension, size, step);
  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

/* is contiguous? [a bit like in TnXIterator] */
static int torch_Tensor_(isContiguous)(lua_State *L)
{
  THTensor *tensor = getTensor(L, 1);
  lua_pushboolean(L, THTensor_(isContiguous)(cltorch_getstate(L), tensor));
  return 1;
}

static int torch_Tensor_(isSameSizeAs)(lua_State *L)
{
  THTensor *self = getTensor(L, 1);
  THTensor *src = getTensor(L, 2);
  lua_pushboolean(L, THTensor_(isSameSizeAs)(cltorch_getstate(L), self, src));
  return 1;
}

static int torch_Tensor_(nElement)(lua_State *L)
{
  THTensor *tensor = getTensor(L, 1);
  lua_pushnumber(L, THTensor_(nElement)(cltorch_getstate(L), tensor));
  return 1;
}

static int torch_Tensor_(copy)(lua_State *L)
{
  THClState *state = cltorch_getstate(L);
  THTensor *tensor = getTensor(L, 1);
  void *src;
  if( (src = luaT_toudata(L, 2, torch_Tensor)) )
    THTensor_(copy)(state, tensor, (THTensor *)src);
  else if( (src = luaT_toudata(L, 2, "torch.ByteTensor")) )
    THTensor_(copyByte)(state, tensor, (THByteTensor *)src);
  else if( (src = luaT_toudata(L, 2, "torch.CharTensor")) )
    THTensor_(copyChar)(state, tensor, (THCharTensor *)src);
  else if( (src = luaT_toudata(L, 2, "torch.ShortTensor")) )
    THTensor_(copyShort)(state, tensor, (THShortTensor *)src);
  else if( (src = luaT_toudata(L, 2, "torch.IntTensor")) )
    THTensor_(copyInt)(state, tensor, (THIntTensor *)src);
  else if( (src = luaT_toudata(L, 2, "torch.LongTensor")) )
    THTensor_(copyLong)(state, tensor, (THLongTensor *)src);
  else if( (src = luaT_toudata(L, 2, "torch.FloatTensor")) )
    THTensor_(copyFloat)(state, tensor, (THFloatTensor *)src);
  else if( (src = luaT_toudata(L, 2, "torch.DoubleTensor")) )
    THTensor_(copyDouble)(state, tensor, (THDoubleTensor *)src);
  else
    luaL_typerror(L, 2, "torch.*Tensor");
  lua_settop(L, 1);
  return 1;
}

static int torch_Tensor_(__newindex__)(lua_State *L)
{
  try {
    THClState *state = cltorch_getstate(L);
    THTensor *tensor = getTensor(L, 1);
    THLongStorage *idx = NULL;
    THByteTensor *mask;
    THClTensor *maskCl;

    if(lua_isnumber(L, 2))
    {
      void *src;
      long index = luaL_checklong(L,2)-1;
      luaL_argcheck(L, tensor->nDimension > 0, 1, "empty tensor");
      if (index < 0) index = tensor->size[0] + index + 1;

      if (lua_isnumber(L,3)) {
        real value = (real)luaL_checknumber(L,3);
        if (tensor->nDimension == 1) {
          //THError("Please copy to cpu, using :float(), then set the value, then copy back using :cl()");
          luaL_argcheck(L, index >= 0 && index < tensor->size[0], 2, "out of range");
          THStorage_(set)(state, tensor->storage, tensor->storageOffset+index*tensor->stride[0], value);
        } else {
          tensor = THTensor_(newWithTensor)(state, tensor);
          THTensor_(narrow)(state, tensor, NULL, 0, index, 1);
          THTensor_(fill)(state, tensor, value);
          THTensor_(free)(state, tensor);
        }
      } else if( (src = getTensorNoCheck(L, 3)) ) {
        tensor = THTensor_(newWithTensor)(state, tensor);
        THTensor_(narrow)(state, tensor, NULL, 0, index, 1);
        THTensor_(copy)(state, tensor, (THTensor *)src);
        THTensor_(free)(state, tensor);
      } else if( (src = luaT_toudata(L, 3, "torch.ByteTensor")) ) {
        tensor = THTensor_(newWithTensor)(state, tensor);
        THTensor_(narrow)(state, tensor, NULL, 0, index, 1);
        THTensor_(copyByte)(state, tensor, (THByteTensor *)src);
        THTensor_(free)(state, tensor);
      } else if( (src = luaT_toudata(L, 3, "torch.CharTensor")) ) {
        tensor = THTensor_(newWithTensor)(state, tensor);
        THTensor_(narrow)(state, tensor, NULL, 0, index, 1);
        THTensor_(copyChar)(state, tensor, (THCharTensor *)src);
        THTensor_(free)(state, tensor);
      } else if( (src = luaT_toudata(L, 3, "torch.ShortTensor")) ) {
        tensor = THTensor_(newWithTensor)(state, tensor);
        THTensor_(narrow)(state, tensor, NULL, 0, index, 1);
        THTensor_(copyShort)(state, tensor, (THShortTensor *)src);
        THTensor_(free)(state, tensor);
      } else if( (src = luaT_toudata(L, 3, "torch.IntTensor")) ) {
        tensor = THTensor_(newWithTensor)(state, tensor);
        THTensor_(narrow)(state, tensor, NULL, 0, index, 1);
        THTensor_(copyInt)(state, tensor, (THIntTensor *)src);
        THTensor_(free)(state, tensor);
      } else if( (src = getLongTensorNoCheck(L, 3)) ) {
        tensor = THTensor_(newWithTensor)(state, tensor);
        THTensor_(narrow)(state, tensor, NULL, 0, index, 1);
        THTensor_(copyLong)(state, tensor, (THLongTensor *)src);
        THTensor_(free)(state, tensor);
      } else if( (src = luaT_toudata(L, 3, "torch.FloatTensor")) ) {
        tensor = THTensor_(newWithTensor)(state, tensor);
        THTensor_(narrow)(state, tensor, NULL, 0, index, 1);
        THTensor_(copyFloat)(state, tensor, (THFloatTensor *)src);
        THTensor_(free)(state, tensor);
      } else if( (src = luaT_toudata(L, 3, "torch.DoubleTensor")) ) {
        tensor = THTensor_(newWithTensor)(state, tensor);
        THTensor_(narrow)(state, tensor, NULL, 0, index, 1);
        THTensor_(copyDouble)(state, tensor, (THDoubleTensor *)src);
        THTensor_(free)(state, tensor);
      } else {
        luaL_typerror(L, 3, "torch.*Tensor");
      }
      lua_pushboolean(L, 1);
    }
    else if((idx = getLongStorageNoCheck(L, 2)))
    {
     // THError("Please copy to cpu, using :float(), then set the value, then copy back using :cl()");
      long index = THTensor_(storageOffset)(state, tensor);
      real value = (real)luaL_checknumber(L,3);
      int dim;

      luaL_argcheck(L, idx->size == tensor->nDimension, 2, "invalid size");

      for(dim = 0; dim < idx->size; dim++)
      {
        long z = idx->data[dim]-1;
        if (z < 0) z = tensor->size[dim] + z + 1;
        luaL_argcheck(L, (z >= 0) && (z < tensor->size[dim]), 2, "index out of bound");
        index += z*tensor->stride[dim];
      }

      THStorage_(set)(state, tensor->storage, index, value);
      lua_pushboolean(L, 1);
      lua_pushboolean(L,0);
    }
    else if(lua_istable(L, 2))
    {
      int dim;
      int cdim = 0;
      int ndims;
      int done = 0;
      ndims = tensor->nDimension;
      luaL_argcheck(L, (long)lua_objlen(L, 2) <= ndims, 2, "too many indices provided");
      tensor = THTensor_(newWithTensor)(state, tensor);
      for(dim = 0; dim < ndims; dim++)
      {
        lua_rawgeti(L, 2, dim+1);
        if(lua_isnumber(L, -1))
        {
          long z = lua_tonumber(L, -1)-1;
          lua_pop(L, 1);
          if (z < 0) z = tensor->size[cdim] + z + 1;
          luaL_argcheck(L, (z >= 0) && (z < tensor->size[cdim]), 2, "index out of bound");
          if(tensor->nDimension == 1) {
            //THError("Please copy to FloatTensor, using :float(), set the value, then copy back using :cl().");
            real value = (real)luaL_checknumber(L,3);
            done = 1;
            THStorage_(set)(state, tensor->storage, tensor->storageOffset+z*tensor->stride[0], value);
          } else {
            THTensor_(select)(state, tensor, NULL, cdim, z);
          }
        }
        else if (lua_istable(L, -1))
        {
          long start = 0;
          long end = tensor->size[cdim]-1;
          lua_rawgeti(L, -1, 1);
          if(lua_isnumber(L, -1)) {
            start = lua_tonumber(L, -1)-1;
            end = start;
          }
          lua_pop(L, 1);
          if (start < 0) start = tensor->size[cdim] + start + 1;
          luaL_argcheck(L, (start >= 0) && (start < tensor->size[cdim]), 2, "start index out of bound");

          lua_rawgeti(L, -1, 2);
          if(lua_isnumber(L, -1)) {
            end = lua_tonumber(L, -1)-1;
          }
          lua_pop(L, 2);
          if (end < 0) end = tensor->size[cdim] + end + 1;
          luaL_argcheck(L, (end >= 0) && (end < tensor->size[cdim]), 2, "end index out of bound");

          luaL_argcheck(L, (end >= start), 2, "end index must be greater or equal to start index");

          THTensor_(narrow)(state, tensor, NULL, cdim++, start, end-start+1);
        }
        else
        {
          break;
        }
      }
      if(!done) {
        /* doing a copy */
        void *src;
        if (lua_isnumber(L,3)) {
          THTensor_(fill)(state, tensor, lua_tonumber(L,3));
        } else if( (src = getTensorNoCheck(L, 3)) ) {
          THTensor_(copy)(state, tensor, (THTensor *)src);
        } else if( (src = luaT_toudata(L, 3, "torch.ByteTensor")) ) {
          THTensor_(copyByte)(state, tensor, (THByteTensor *)src);
        } else if( (src = luaT_toudata(L, 3, "torch.CharTensor")) ) {
          THTensor_(copyChar)(state, tensor, (THCharTensor *)src);
        } else if( (src = luaT_toudata(L, 3, "torch.ShortTensor")) ) {
          THTensor_(copyShort)(state, tensor, (THShortTensor *)src);
        } else if( (src = luaT_toudata(L, 3, "torch.IntTensor")) ) {
          THTensor_(copyInt)(state, tensor, (THIntTensor *)src);
        } else if( (src = getLongTensorNoCheck(L, 3)) ) {
          THTensor_(copyLong)(state, tensor, (THLongTensor *)src);
        } else if( (src = luaT_toudata(L, 3, "torch.FloatTensor")) ) {
          THTensor_(copyFloat)(state, tensor, (THFloatTensor *)src);
        } else if( (src = luaT_toudata(L, 3, "torch.DoubleTensor")) ) {
          THTensor_(copyDouble)(state, tensor, (THDoubleTensor *)src);
        } else {
          luaL_typerror(L, 3, "torch.*Tensor");
        }
      }
      THTensor_(free)(state, tensor);
      lua_pushboolean(L, 1);
    }
    else if((mask = getByteTensorNoCheck(L, 2)))
    {
      THTensor *vals;
      if (lua_isnumber(L, 3))
      {
        THTensor_(maskedFillByte)(state, tensor, mask,
                                  (real)(luaL_checknumber(L,3)));
      }
      else if((vals = getTensorNoCheck(L, 3)))
      {
        THTensor_(maskedCopyByte)(state, tensor, mask, vals);
      }
      else
      {
        luaL_error(L,"number or tensor expected");
      }
    }
    else if((maskCl = getClTensorNoCheck(L, 2)))
    {
      THTensor *vals;
      if (lua_isnumber(L, 3))
      {
        THTensor_(maskedFill)(state, tensor, maskCl,
                              (real)(luaL_checknumber(L,3)));
      }
      else if((vals = getTensorNoCheck(L, 3)))
      {
        THTensor_(maskedCopy)(state, tensor, maskCl, vals);
      }
      else
      {
        luaL_error(L,"number or tensor expected");
      }
    }
    else
      lua_pushboolean(L, 0);

    return 1;
  } catch(exception &e) {
    THError("Something went wrong: %s", e.what());
    return 0;
  }
}

static int torch_Tensor_(__index__)(lua_State *L)
{
  try {
    THClState *state = cltorch_getstate(L);
    THTensor *tensor = getTensor(L, 1);
    THLongStorage *idx = NULL;
    THByteTensor *mask;
    THClTensor *maskCl;

    if(lua_isnumber(L, 2))
    {
      long index = luaL_checklong(L,2)-1;

      luaL_argcheck(L, tensor->nDimension > 0, 1, "empty tensor");
      if (index < 0) index = tensor->size[0] + index + 1;
      luaL_argcheck(L, index >= 0 && index < tensor->size[0], 2, "out of range");

      if(tensor->nDimension == 1)
      {
          //THError("Please copy to FloatTensor, using :float(), then get the value.");
        lua_pushnumber(L, THStorage_(get)(state, tensor->storage, tensor->storageOffset+index*tensor->stride[0]));
      }
      else
      {
        tensor = THTensor_(newWithTensor)(state, tensor);
        THTensor_(select)(state, tensor, NULL, 0, index);
        luaT_pushudata(L, tensor, torch_Tensor);
      }
      lua_pushboolean(L, 1);
      return 2;
    }
    else if((idx = getLongStorageNoCheck(L, 2)))
    {
  //    THError("Please copy to FloatTensor, using :float(), then read the value");
      long index = THTensor_(storageOffset)(state, tensor);
      int dim;

      luaL_argcheck(L, idx->size == tensor->nDimension, 2, "invalid size");

      for(dim = 0; dim < idx->size; dim++)
      {
        long z = idx->data[dim]-1;
        if (z < 0) z = tensor->size[dim] + z + 1;
        luaL_argcheck(L, (z >= 0) && (z < tensor->size[dim]), 2, "index out of bound");
        index += z*tensor->stride[dim];
      }
      lua_pushnumber(L, (double)THStorage_(get)(state, THTensor_(storage)(state, tensor), index));
      lua_pushboolean(L, 1);
      return 2;
  //    lua_pushboolean(L,0);
   //   return 1;
    }
    else if(lua_istable(L, 2))
    {
      int dim;
      int cdim = 0;
      int ndims;
      int done = 0;

      ndims = tensor->nDimension;
      luaL_argcheck(L, (long)lua_objlen(L, 2) <= ndims, 2, "too many indices provided");
      tensor = THTensor_(newWithTensor)(state, tensor);

      for(dim = 0; dim < ndims; dim++)
      {
        lua_rawgeti(L, 2, dim+1);
        if(lua_isnumber(L, -1))
        {
          long z = lua_tonumber(L, -1)-1;
          lua_pop(L, 1);
          if (z < 0) z = tensor->size[cdim] + z + 1;
          luaL_argcheck(L, (z >= 0) && (z < tensor->size[cdim]), 2, "index out of bound");
          if(tensor->nDimension == 1) {
          //THError("Please copy to cpu, using :float(), then get the value");
            done = 1;
            lua_pushnumber(L, THStorage_(get)(state, tensor->storage, tensor->storageOffset+z*tensor->stride[0]));
          } else {
            THTensor_(select)(state, tensor, NULL, cdim, z);
          }
        }
        else if (lua_istable(L, -1))
        {
          long start = 0;
          long end = tensor->size[cdim]-1;
          lua_rawgeti(L, -1, 1);
          if(lua_isnumber(L, -1)) {
            start = lua_tonumber(L, -1)-1;
            end = start;
          }
          lua_pop(L, 1);
          if (start < 0) start = tensor->size[cdim] + start + 1;
          luaL_argcheck(L, (start >= 0) && (start < tensor->size[cdim]), 2, "start index out of bound");

          lua_rawgeti(L, -1, 2);
          if(lua_isnumber(L, -1)) {
            end = lua_tonumber(L, -1)-1;
          }
          lua_pop(L, 2);
          if (end < 0) end = tensor->size[cdim] + end + 1;
          luaL_argcheck(L, (end >= 0) && (end < tensor->size[cdim]), 2, "end index out of bound");

          luaL_argcheck(L, (end >= start), 2, "end index must be greater or equal to start index");

          THTensor_(narrow)(state, tensor, NULL, cdim++, start, end-start+1);
        }
        else
        {
          break;
        }
      }
      if(!done) {
        luaT_pushudata(L, tensor, torch_Tensor);
      } else {
        THTensor_(free)(state, tensor);
      }
      lua_pushboolean(L, 1);
      return 2;
    }
    else if((mask = getByteTensorNoCheck(L, 2)))
    {
      THTensor *vals = THTensor_(newv2)(state, tensor->storage->device);
      THTensor_(maskedSelectByte)(state, vals, tensor, mask);
      luaT_pushudata(L, vals, torch_Tensor);
      lua_pushboolean(L, 1);
      return 2;
    }
    else if((maskCl = getClTensorNoCheck(L, 2)))
    {
      THTensor *vals = THTensor_(newv2)(state, tensor->storage->device);
      THTensor_(maskedSelect)(state, vals, tensor, maskCl);
      luaT_pushudata(L, vals, torch_Tensor);
      lua_pushboolean(L, 1);
      return 2;
    }
    else
    {
      lua_pushboolean(L, 0);
      return 1;
    }
  } catch(exception &e) {
    THError("Something went wrong: %s", e.what());
    return 0;
  }
}

static int torch_Tensor_(retain)(lua_State *L)
{
  THTensor *tensor = getTensor(L, 1);
  THTensor_(retain)(cltorch_getstate(L), tensor);
  return 0;
}

static int torch_Tensor_(free)(lua_State *L)
{
  THTensor *tensor = getTensor(L, 1);
  THTensor_(free)(cltorch_getstate(L), tensor);
  return 0;
}

/* helpful functions */
static void torch_Tensor_(c_readSizeStride)(lua_State *L, int index, int allowStride, THLongStorage **size_, THLongStorage **stride_)
{
  THLongStorage *size = NULL;
  THLongStorage *stride = NULL;

  if((size = getLongStorageNoCheck(L, index)))
  {
    if(!lua_isnoneornil(L, index+1))
    {
      if( (stride = getLongStorageNoCheck(L, index+1)) )
        luaL_argcheck(L, stride->size == size->size, index+1, "provided stride and size are inconsistent");
      else
        luaL_argcheck(L, 0, index+1, "torch.LongStorage expected");
    }
    THLongStorage_retain(size);
    if(stride)
      THLongStorage_retain(stride);
  }
  else
  {
    int i;

    size = THLongStorage_newWithSize(8);
    stride = THLongStorage_newWithSize(8);
    THLongStorage_fill(size, -1);
    THLongStorage_fill(stride, -1);

    if(allowStride)
    {
      for(i = 0; i < 8; i++)
      {
        if(lua_isnone(L, index+2*i))
          break;
        size->data[i] = luaL_checklong(L, index+2*i);

        if(lua_isnone(L, index+2*i+1))
          break;
        stride->data[i] = luaL_checklong(L, index+2*i+1);
      }
    }
    else
    {
      for(i = 0; i < 8; i++)
      {
        if(lua_isnone(L, index+i))
          break;
        size->data[i] = luaL_checklong(L, index+i);
      }
    }
  }

  *size_ = size;
  *stride_ = stride;
}

static void torch_Tensor_(c_readTensorStorageSizeStride)(lua_State *L, int index, int allowNone, int allowTensor, int allowStorage, int allowStride,
                                                         THStorage **storage_, long *storageOffset_, THLongStorage **size_, THLongStorage **stride_)
{
  THClState *state = cltorch_getstate(L);
  THTensor *src = NULL;
  THStorage *storage = NULL;

  int arg1Type = lua_type(L, index);

  if( allowNone && (arg1Type == LUA_TNONE) )
  {
    *storage_ = NULL;
    *storageOffset_ = 0;
    *size_ = NULL;
    *stride_ = NULL;
    return;
  }
  else if( allowTensor && (arg1Type == LUA_TUSERDATA) && (src = getTensorNoCheck(L, index)) )
  {
    *storage_ = src->storage;
    *storageOffset_ = src->storageOffset;
    *size_ = THTensor_(newSizeOf)(state, src);
    *stride_ = THTensor_(newStrideOf)(state, src);
    return;
  }
  else if( allowStorage && (arg1Type == LUA_TUSERDATA) && (storage = getStorageNoCheck(L, index)) )
  {
    *storage_ = storage;
    if(lua_isnone(L, index+1))
    {
      *storageOffset_ = 0;
      *size_ = THLongStorage_newWithSize1(storage->size);
      *stride_ = THLongStorage_newWithSize1(1);
    }
    else
    {
      *storageOffset_ = luaL_checklong(L, index+1)-1;
      torch_Tensor_(c_readSizeStride)(L, index+2, allowStride, size_, stride_);
    }
    return;
  }
  else if( (arg1Type == LUA_TNUMBER) || (luaT_toudata(L, index, "torch.LongStorage")) )
  {
    *storage_ = NULL;
    *storageOffset_ = 0;
    torch_Tensor_(c_readSizeStride)(L, index, 0, size_, stride_);

    return;
  }

  *storage_ = NULL;
  *storageOffset_ = 0;

  if(allowTensor && allowStorage)
      luaL_argcheck(L, 0, index, "expecting number or Tensor or Storage");
  else if(allowTensor)
      luaL_argcheck(L, 0, index, "expecting number or Tensor");
  else if(allowStorage)
      luaL_argcheck(L, 0, index, "expecting number or Storage");
  else
      luaL_argcheck(L, 0, index, "expecting number");
}

static int torch_Tensor_(factory)(lua_State *L)
{
  THTensor *tensor = THTensor_(newv2)(cltorch_getstate(L), cltorch_getstate(L)->currentDevice);
  luaT_pushudata(L, tensor, torch_Tensor);
  return 1;
}

static int torch_Tensor_(write)(lua_State *L)
{
  try {
    THTensor *tensor = getTensor(L, 1);
    THFile *file = (THFile *)luaT_checkudata(L, 2, "torch.File");

    THFile_writeIntScalar(file, tensor->nDimension);
    THFile_writeLongRaw(file, tensor->size, tensor->nDimension);
    THFile_writeLongRaw(file, tensor->stride, tensor->nDimension);
    THFile_writeLongScalar(file, tensor->storageOffset+1); /* to respect Lua convention */

    lua_getfield(L, 2, "writeObject"); /* the method */
    lua_pushvalue(L, 2); /* the file */
    /* the storage */
    if(tensor->storage)
    {
      THStorage_(retain)(cltorch_getstate(L), tensor->storage);
      luaT_pushudata(L, tensor->storage, torch_Storage);
    }
    else
      lua_pushnil(L);

    lua_call(L, 2, 0); /* call the method */
  } catch(exception &e) {
    THError("Something went wrong: %s", e.what());
  }

  return 0;
}

static int torch_Tensor_(read)(lua_State *L)
{
  try {
    THTensor *tensor = getTensor(L, 1);
    THFile *file = (THFile *)luaT_checkudata(L, 2, "torch.File");

    tensor->nDimension = THFile_readIntScalar(file);
    tensor->size = (long int *)THAlloc(sizeof(long)*tensor->nDimension);
    tensor->stride = (long int *)THAlloc(sizeof(long)*tensor->nDimension);
    THFile_readLongRaw(file, tensor->size, tensor->nDimension);
    THFile_readLongRaw(file, tensor->stride, tensor->nDimension);
    tensor->storageOffset = THFile_readLongScalar(file);
    tensor->storageOffset--;  /* to respect Lua convention */

    lua_getfield(L, 2, "readObject"); /* the method */
    lua_pushvalue(L, 2); /* the file */
    lua_call(L, 1, 1); /* call the method */

    tensor->storage = getStorageNoCheck(L, -1);
    if(tensor->storage)
      THStorage_(retain)(cltorch_getstate(L), tensor->storage);
  } catch(exception &e) {
    THError("Something went wrong: %s", e.what());
  }

  return 0;
}

static const struct luaL_Reg torch_Tensor_(_) [] = {
  {"retain", torch_Tensor_(retain)},
  {"free", torch_Tensor_(free)},
  {"contiguous", torch_Tensor_(contiguous)},
  {"size", torch_Tensor_(size)},
  {"__len__", torch_Tensor_(size)},
  {"stride", torch_Tensor_(stride)},
  {"dim", torch_Tensor_(nDimension)},
  {"nDimension", torch_Tensor_(nDimension)},
  {"set", torch_Tensor_(set)},
  {"storage", torch_Tensor_(storage)},
  {"storageOffset", torch_Tensor_(storageOffset)},
  {"clone", torch_Tensor_(clone)},
  {"contiguous", torch_Tensor_(contiguous)},
  {"resizeAs", torch_Tensor_(resizeAs)},
//  {"gather", torch_Tensor_(gather)},
  {"resize", torch_Tensor_(resize)},
  {"narrow", torch_Tensor_(narrow)},
  {"sub", torch_Tensor_(sub)},
  {"select", torch_Tensor_(select)},
  {"index", torch_Tensor_(indexSelect)},
  {"indexCopy", torch_Tensor_(indexCopy)},
//  {"asFloat", torch_Tensor_(asFloat)},
  {"indexFill", torch_Tensor_(indexFill)},
  {"transpose", torch_Tensor_(transpose)},
  {"t", torch_Tensor_(t)},
  {"unfold", torch_Tensor_(unfold)},
  {"isContiguous", torch_Tensor_(isContiguous)},
  {"isSameSizeAs", torch_Tensor_(isSameSizeAs)},
  {"nElement", torch_Tensor_(nElement)},
  {"copy", torch_Tensor_(copy)},
//  {"apply", torch_Tensor_(apply)},
//  {"map", torch_Tensor_(map)},
//  {"map2", torch_Tensor_(map2)},
  {"read", torch_Tensor_(read)},
  {"write", torch_Tensor_(write)},
  {"__index__", torch_Tensor_(__index__)},
  {"__newindex__", torch_Tensor_(__newindex__)},
  {NULL, NULL}
};

void torch_Tensor_(init)(lua_State *L)
{
  luaT_newmetatable(L, torch_Tensor, NULL,
                    torch_Tensor_(new), torch_Tensor_(free), torch_Tensor_(factory));
  luaL_setfuncs(L, torch_Tensor_(_), 0);
  lua_pop(L, 1);
}

#endif

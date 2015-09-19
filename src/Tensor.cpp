#include "torch/utils.h"
#include "THCl.h"
#include "THFile.h"
#include "luaT.h"

extern "C" {
  void cltorch_ClTensor_init(lua_State *L);
}

/* everything is as the generic Storage.c, except few things (see below) */

#define real float
#define Real Cl

#define torch_Storage_(NAME) TH_CONCAT_4(torch_,Real,Storage_,NAME)
#define torch_Storage TH_CONCAT_STRING_3(torch.,Real,Storage)
#define torch_Tensor_(NAME) TH_CONCAT_4(torch_,Real,Tensor_,NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)

#define TH_GENERIC_FILE "generic/Tensor.c"
#include "generic/Tensor.cpp"
#undef TH_GENERIC_FILE

#undef real
#undef Real

/* now we overwrite some methods specific to ClTensor */
static int cltorch_ClTensor_copy(lua_State *L)
{
  THClState *state = cltorch_getstate(L);
  THClTensor *storage = (THClTensor *)luaT_checkudata(L, 1, "torch.ClTensor");
  void *src;
  if( (src = luaT_toudata(L, 2, "torch.ClTensor")) )
    THClTensor_copy(state, storage, (THClTensor *)src);
  else if( (src = luaT_toudata(L, 2, "torch.ByteTensor")) )
    THClTensor_copyByte(state, storage, (THByteTensor *)src);
  else if( (src = luaT_toudata(L, 2, "torch.CharTensor")) )
    THClTensor_copyChar(state, storage, (THCharTensor *)src);
  else if( (src = luaT_toudata(L, 2, "torch.ShortTensor")) )
    THClTensor_copyShort(state, storage, (THShortTensor *)src);
  else if( (src = luaT_toudata(L, 2, "torch.IntTensor")) )
    THClTensor_copyInt(state, storage, (THIntTensor *)src);
  else if( (src = luaT_toudata(L, 2, "torch.LongTensor")) )
    THClTensor_copyLong(state, storage, (THLongTensor *)src);
  else if( (src = luaT_toudata(L, 2, "torch.FloatTensor")) )
    THClTensor_copyFloat(state, storage, (THFloatTensor *)src);
  else if( (src = luaT_toudata(L, 2, "torch.DoubleTensor")) )
    THClTensor_copyDouble(state, storage, (THDoubleTensor *)src);
  else if( (src = luaT_toudata(L, 2, "torch.ClTensor")) )
    THClTensor_copyCl(state, storage, (THClTensor *)src);
  else
    luaL_typerror(L, 2, "torch.*Tensor");

  lua_settop(L, 1);
  return 1;
}

#define CL_IMPLEMENT_TENSOR_COPY(TYPEC)                               \
  static int cltorch_##TYPEC##Tensor_copy(lua_State *L)                 \
  {                                                                     \
    TH##TYPEC##Tensor *storage = (TH##TYPEC##Tensor *)luaT_checkudata(L, 1, "torch." #TYPEC "Tensor"); \
    void *src;                                                          \
    if( (src = luaT_toudata(L, 2, "torch." #TYPEC "Tensor")) )          \
      TH##TYPEC##Tensor_copy(storage, (TH##TYPEC##Tensor *)src);                             \
    else if( (src = luaT_toudata(L, 2, "torch.ByteTensor")) )           \
      TH##TYPEC##Tensor_copyByte(storage, (THByteTensor *)src);                         \
    else if( (src = luaT_toudata(L, 2, "torch.CharTensor")) )           \
      TH##TYPEC##Tensor_copyChar(storage, (THCharTensor *)src);                         \
    else if( (src = luaT_toudata(L, 2, "torch.ShortTensor")) )          \
      TH##TYPEC##Tensor_copyShort(storage, (THShortTensor *)src);                        \
    else if( (src = luaT_toudata(L, 2, "torch.IntTensor")) )            \
      TH##TYPEC##Tensor_copyInt(storage, (THIntTensor *)src);                          \
    else if( (src = luaT_toudata(L, 2, "torch.LongTensor")) )           \
      TH##TYPEC##Tensor_copyLong(storage, (THLongTensor *)src);                         \
    else if( (src = luaT_toudata(L, 2, "torch.FloatTensor")) )          \
      TH##TYPEC##Tensor_copyFloat(storage, (THFloatTensor *)src);                        \
    else if( (src = luaT_toudata(L, 2, "torch.DoubleTensor")) )         \
      TH##TYPEC##Tensor_copyDouble(storage, (THDoubleTensor *)src);                       \
    else if( (src = luaT_toudata(L, 2, "torch.ClTensor")) )           \
      TH##TYPEC##Tensor_copyCl(cltorch_getstate(L), storage, (THClTensor *)src);    \
    else                                                                \
      luaL_typerror(L, 2, "torch.*Tensor");                             \
                                                                        \
    lua_settop(L, 1);                                                   \
    return 1;                                                           \
  }

CL_IMPLEMENT_TENSOR_COPY(Byte)
CL_IMPLEMENT_TENSOR_COPY(Char)
CL_IMPLEMENT_TENSOR_COPY(Short)
CL_IMPLEMENT_TENSOR_COPY(Int)
CL_IMPLEMENT_TENSOR_COPY(Long)
CL_IMPLEMENT_TENSOR_COPY(Float)
CL_IMPLEMENT_TENSOR_COPY(Double)

static void THFloatTensor_computesz(THFloatTensor *self, long **sz_, long **st_)
{
  long *sz, *st, *szh;
  int i;

  sz = (long*)THAlloc(sizeof(long)*self->nDimension);
  st = (long*)THAlloc(sizeof(long)*self->nDimension);
  szh = (long*)THAlloc(sizeof(long)*self->nDimension);

  for(i = self->nDimension-1; i >= 0; i--)
  {
    if(i == self->nDimension-1)
      szh[i] = 1;
    else
      szh[i] = szh[i+1]*self->size[i+1];
  }

  memcpy(sz, szh, self->nDimension * sizeof(long));
  memcpy(st, self->stride, self->nDimension * sizeof(long));
  THFree(szh);

  *sz_ = sz;
  *st_ = st;
}

void THFloatTensor_kernel_copy(float *dst,
                                         long *dst_sz, long *dst_st, int dst_dim,
                                         float *src,
                                         long *src_sz, long *src_st, int src_dim,
                                         long n_elem)
{
  long k;

  for(k = 0; k < n_elem; k++)
  {
    long src_idx = 0;
    long src_rest = k;
    long dst_idx = 0;
    long dst_rest = k;
    int dim;

    for(dim = 0; dim < dst_dim; dim++)
    {
      dst_idx += (dst_rest/dst_sz[dim])*dst_st[dim];
      dst_rest = dst_rest % dst_sz[dim];
    }

    for(dim = 0; dim < src_dim; dim++)
    {
      src_idx += (src_rest/src_sz[dim])*src_st[dim];
      src_rest = src_rest % src_sz[dim];
    }

    dst[dst_idx] = src[src_idx];
  }
}

static int cl_FloatTensor_fakecopy(lua_State *L)
{
  THFloatTensor *self = (THFloatTensor *)luaT_checkudata(L, 1, "torch.FloatTensor");
  THFloatTensor *src = (THFloatTensor *)luaT_checkudata(L, 2, "torch.FloatTensor");
  long *d_self_sz, *d_self_st, *d_src_sz, *d_src_st;
  long nElement = THFloatTensor_nElement(self);

  THArgCheck(THFloatTensor_nElement(self) == THFloatTensor_nElement(src), 2, "sizes do not match");

  THFloatTensor_computesz(self, &d_self_sz, &d_self_st);
  THFloatTensor_computesz(src, &d_src_sz, &d_src_st);

  THFloatTensor_kernel_copy(THFloatTensor_data(self),
                            d_self_sz, d_self_st, self->nDimension,
                            THFloatTensor_data(src),
                            d_src_sz, d_src_st, src->nDimension,
                            nElement);

  THFree(d_self_sz);
  THFree(d_self_st);
  THFree(d_src_sz);
  THFree(d_src_st);

  lua_settop(L, 1);
  return 1;
}

static int cltorch_ClTensor_getDevice(lua_State *L) {
  THClTensor *tensor = (THClTensor *)luaT_checkudata(L, 1, "torch.ClTensor");
  lua_pushinteger(L, THClTensor_getDevice(cltorch_getstate(L), tensor) + 1);
  return 1;
}

void cltorch_ClTensor_init(lua_State* L)
{
  /* the standard stuff */
  torch_ClTensor_init(L);

  /* additional methods */
  luaT_pushmetatable(L, "torch.FloatTensor");
  lua_pushcfunction(L, cl_FloatTensor_fakecopy);
  lua_setfield(L, -2, "fakecopy");
  lua_pop(L, 1);

  /* the copy methods */
  {
    int i;

    const char* tnames[8] = {"torch.ByteTensor",
                             "torch.CharTensor",
                             "torch.ShortTensor",
                             "torch.IntTensor",
                             "torch.LongTensor",
                             "torch.FloatTensor",
                             "torch.DoubleTensor",
                             "torch.ClTensor"};

    static int (*funcs[8])(lua_State*) = {cltorch_ByteTensor_copy,
                                          cltorch_CharTensor_copy,
                                          cltorch_ShortTensor_copy,
                                          cltorch_IntTensor_copy,
                                          cltorch_LongTensor_copy,
                                          cltorch_FloatTensor_copy,
                                          cltorch_DoubleTensor_copy,
                                          cltorch_ClTensor_copy};

    for(i = 0; i < 8; i++)
    {
      luaT_pushmetatable(L, tnames[i]);
      lua_pushcfunction(L, funcs[i]);
      lua_setfield(L, -2, "copy");
      lua_pop(L, 1);
    }
  }

  luaT_pushmetatable(L, "torch.ClTensor");
  lua_pushcfunction(L, cltorch_ClTensor_getDevice);
  lua_setfield(L, -2, "getDevice");

  lua_pop(L, 1);
}


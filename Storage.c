#include "torch/utils.h"
#include "THCl.h"
//#include "THFile.h"
#include "luaT.h"

/* everything is as the generic Storage.c, except few things (see below) */

#define real float
#define Real Cl
#define TH_GENERIC_FILE "generic/Storage.c"

#define torch_Storage_(NAME) TH_CONCAT_4(torch_,Real,Storage_,NAME)

#define THFile_readRealRaw(file, data, size)                            \
  {                                                                     \
    float *fdata = (float*)THAlloc(sizeof(float)*size);                 \
    THFile_readFloatRaw(file, fdata, size);                             \
/*    THClCheck(clMemcpy(data, fdata, size * sizeof(float), clMemcpyHostToDevice));*/ \
    THFree(fdata);                                                      \
  }

#define THFile_writeRealRaw(file, data, size)                           \
  {                                                                     \
    float *fdata = (float*)THAlloc(sizeof(float)*size);                 \
/*    THClCheck(clMemcpy(fdata, data, size * sizeof(float), clMemcpyDeviceToHost));*/ \
    THFile_writeFloatRaw(file, fdata, size);                            \
    THFree(fdata);                                                      \
  }

#define torch_Storage TH_CONCAT_STRING_3(torch.,Real,Storage)

#include "generic/Storage.c"

#undef real
#undef Real
#undef TH_GENERIC_FILE

/* now we overwrite some methods specific to ClStorage */

static int cltorch_ClStorage_copy(lua_State *L)
{
  THClState *state = cltorch_getstate(L);
  THClStorage *storage = luaT_checkudata(L, 1, "torch.ClStorage");
  void *src;
  if( (src = luaT_toudata(L, 2, "torch.ClStorage")) )
    THClStorage_copy(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.ByteStorage")) )
    THClStorage_copyByte(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CharStorage")) )
    THClStorage_copyChar(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.ShortStorage")) )
    THClStorage_copyShort(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.IntStorage")) )
    THClStorage_copyInt(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.LongStorage")) )
    THClStorage_copyLong(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.FloatStorage")) )
    THClStorage_copyFloat(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.DoubleStorage")) )
    THClStorage_copyDouble(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.ClStorage")) )
    THClStorage_copyCl(state, storage, src);
  else
    luaL_typerror(L, 2, "torch.*Storage");

  lua_settop(L, 1);
  return 1;
}

#define CL_IMPLEMENT_STORAGE_COPY(TYPEC)                              \
  static int cltorch_##TYPEC##Storage_copy(lua_State *L)                \
  {                                                                     \
    TH##TYPEC##Storage *storage = luaT_checkudata(L, 1, "torch." #TYPEC "Storage"); \
    void *src;                                                          \
    if( (src = luaT_toudata(L, 2, "torch." #TYPEC "Storage")) )         \
      TH##TYPEC##Storage_copy(storage, src);                            \
    else if( (src = luaT_toudata(L, 2, "torch.ByteStorage")) )          \
      TH##TYPEC##Storage_copyByte(storage, src);                        \
    else if( (src = luaT_toudata(L, 2, "torch.CharStorage")) )          \
      TH##TYPEC##Storage_copyChar(storage, src);                        \
    else if( (src = luaT_toudata(L, 2, "torch.ShortStorage")) )         \
      TH##TYPEC##Storage_copyShort(storage, src);                       \
    else if( (src = luaT_toudata(L, 2, "torch.IntStorage")) )           \
      TH##TYPEC##Storage_copyInt(storage, src);                         \
    else if( (src = luaT_toudata(L, 2, "torch.LongStorage")) )          \
      TH##TYPEC##Storage_copyLong(storage, src);                        \
    else if( (src = luaT_toudata(L, 2, "torch.FloatStorage")) )         \
      TH##TYPEC##Storage_copyFloat(storage, src);                       \
    else if( (src = luaT_toudata(L, 2, "torch.DoubleStorage")) )        \
      TH##TYPEC##Storage_copyDouble(storage, src);                      \
    else if( (src = luaT_toudata(L, 2, "torch.ClStorage")) )          \
      TH##TYPEC##Storage_copyCl(cltorch_getstate(L), storage, src);   \
    else                                                                \
      luaL_typerror(L, 2, "torch.*Storage");                            \
                                                                        \
    lua_settop(L, 1);                                                   \
    return 1;                                                           \
}

CL_IMPLEMENT_STORAGE_COPY(Byte)
CL_IMPLEMENT_STORAGE_COPY(Char)
CL_IMPLEMENT_STORAGE_COPY(Short)
CL_IMPLEMENT_STORAGE_COPY(Int)
CL_IMPLEMENT_STORAGE_COPY(Long)
CL_IMPLEMENT_STORAGE_COPY(Float)
CL_IMPLEMENT_STORAGE_COPY(Double)

void cltorch_ClStorage_init(lua_State* L)
{
  /* the standard stuff */
  torch_ClStorage_init(L);

  /* the copy methods */
  {
    int i;

    const void* tnames[8] = {"torch.ByteStorage",
                             "torch.CharStorage",
                             "torch.ShortStorage",
                             "torch.IntStorage",
                             "torch.LongStorage",
                             "torch.FloatStorage",
                             "torch.DoubleStorage",
                             "torch.ClStorage"};

    static int (*funcs[8])(lua_State*) = {cltorch_ByteStorage_copy,
                                          cltorch_CharStorage_copy,
                                          cltorch_ShortStorage_copy,
                                          cltorch_IntStorage_copy,
                                          cltorch_LongStorage_copy,
                                          cltorch_FloatStorage_copy,
                                          cltorch_DoubleStorage_copy,
                                          cltorch_ClStorage_copy};

    for(i = 0; i < 8; i++)
    {
      luaT_pushmetatable(L, tnames[i]);
      lua_pushcfunction(L, funcs[i]);
      lua_setfield(L, -2, "copy");
      lua_pop(L, 1);
    }
  }
}

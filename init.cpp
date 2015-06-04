#include <stdio.h>
#include <iostream>
#include "EasyCL.h"
using namespace std;

//#include "THClTensorRandom.h"

//extern void clnn_ClStorage_init(lua_State* L);
//extern void clnn_ClTensor_init(lua_State* L);
//extern void clnn_ClTensorMath_init(lua_State* L);
//extern void clnn_ClTensorOperator_init(lua_State* L);


extern "C" {
  #include "lua.h"
  #include "utils.h"
  #include "luaT.h"
  #include "THClGeneral.h"
  int luaopen_libclnn( lua_State *L );
}

#define SET_DEVN_PROP(NAME) \
  lua_pushnumber(L, prop.NAME); \
  lua_setfield(L, -2, #NAME);

static int clnn_getDeviceProperties(lua_State *L)
{
  cout << "clnn_getDeviceProperties" << endl;

  lua_newtable(L);

//  size_t freeMem;
//  THCudaCheck(cudaMemGetInfo (&freeMem, NULL));
//  lua_pushnumber(L, freeMem);
//  lua_setfield(L, -2, "freeGlobalMem");

//  lua_pushstring(L, prop.name);
//  lua_setfield(L, -2, "name");

  lua_pushstring(L, "v0.0.1");
  lua_setfield(L, -2, "version");

  return 1;
}

//static int cutorch_getState(lua_State *L)
//{
//  lua_getglobal(L, "cutorch");
//  lua_getfield(L, -1, "_state");
//  lua_remove(L, -2);
//  return 1;
//}

static const struct luaL_Reg clnn_stuff__ [] = {
  {"getDeviceProperties", clnn_getDeviceProperties},
  {NULL, NULL}
};

int luaopen_libclnn( lua_State *L ) {
  printf("luaopen_libclnn called :-)\n");
  cout << " try cout" << endl;

  lua_newtable(L);

  luaL_setfuncs(L, clnn_stuff__, 0);
  cout << "setfuncs done" << endl;

  THClState* state = (THClState*)malloc(sizeof(THClState));
  cout << "allocated THClState" << endl;
  THClInit(state);
  cout << "THClInit done" << endl;

//  cltorch_ClStorage_init(L);
//  cltorch_ClTensor_init(L);
//  cltorch_ClTensorMath_init(L);
//  cltorch_ClTensorOperator_init(L);

  /* Store state in cutorch table. */
  lua_pushlightuserdata(L, state);
  lua_setfield(L, -2, "_state");
  cout << "luaopen_libclnn done\n" << endl;

  return 1;
}


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

namespace clnn {

  #define SET_DEVN_PROP(NAME) \
    lua_pushnumber(L, prop.NAME); \
    lua_setfield(L, -2, #NAME);

  void setProperty(lua_State *L, string name, int value)
  {
    lua_pushnumber(L, value);
    lua_setfield(L, -2, name.c_str());
  }
  void setProperty(lua_State *L, string name, string value)
  {
    lua_pushstring(L, value.c_str());
    lua_setfield(L, -2, name.c_str());
  }

  static int clnn_getDeviceProperties(lua_State *L)
  {
    cout << "clnn_getDeviceProperties" << endl;

    int device = (int)luaL_checknumber(L, 1)-1;
    cout << "device: " << device << endl;

    EasyCL *cl = EasyCL::createForIndexedGpu(device); // probably not most efficient way of doing this, since it creates all the queues and stuff...
                                                      // but easy for now

    lua_newtable(L);

    setProperty(L, "localMemorySize", cl->getLocalMemorySize());
    setProperty(L, "localMemorySizeKB", cl->getLocalMemorySizeKB());
    setProperty(L, "maxAllocSizeMB", cl->getMaxAllocSizeMB());
    setProperty(L, "maxWorkgroupSize", cl->getMaxWorkgroupSize());

    delete cl;

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
}

int luaopen_libclnn( lua_State *L ) {
  printf("luaopen_libclnn called :-)\n");
  cout << " try cout" << endl;

  lua_newtable(L);

  luaL_setfuncs(L, clnn::clnn_stuff__, 0);
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


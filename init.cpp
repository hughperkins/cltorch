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

extern "C" {
  static int clnn_getDeviceProperties(lua_State *L);
}

extern "C" {
static int clnn_getDeviceProperties(lua_State *L)
{
  cout << "clnn_getDeviceProperties" << endl;
//  int device = (int)luaL_checknumber(L, 1)-1;

  // switch context to given device so the call to cudaMemGetInfo is for the correct device
//  int oldDevice;
//  THCudaCheck(cudaGetDevice(&oldDevice));
//  THCudaCheck(cudaSetDevice(device));

//  struct cudaDeviceProp prop;
//  THCudaCheck(cudaGetDeviceProperties(&prop, device));
//  lua_newtable(L);
//  SET_DEVN_PROP(canMapHostMemory);
//  SET_DEVN_PROP(clockRate);
//  SET_DEVN_PROP(computeMode);
//  SET_DEVN_PROP(deviceOverlap);
//  SET_DEVN_PROP(integrated);
//  SET_DEVN_PROP(kernelExecTimeoutEnabled);
//  SET_DEVN_PROP(major);
//  SET_DEVN_PROP(maxThreadsPerBlock);
//  SET_DEVN_PROP(memPitch);
//  SET_DEVN_PROP(minor);
//  SET_DEVN_PROP(multiProcessorCount);
//  SET_DEVN_PROP(regsPerBlock);
//  SET_DEVN_PROP(sharedMemPerBlock);
//  SET_DEVN_PROP(textureAlignment);
//  SET_DEVN_PROP(totalConstMem);
//  SET_DEVN_PROP(totalGlobalMem);
//  SET_DEVN_PROP(warpSize);
//  SET_DEVN_PROP(pciBusID);
//  SET_DEVN_PROP(pciDeviceID);
//  SET_DEVN_PROP(pciDomainID);
//  SET_DEVN_PROP(maxTexture1D);
//  SET_DEVN_PROP(maxTexture1DLinear);

//  size_t freeMem;
//  THCudaCheck(cudaMemGetInfo (&freeMem, NULL));
//  lua_pushnumber(L, freeMem);
//  lua_setfield(L, -2, "freeGlobalMem");

//  lua_pushstring(L, prop.name);
//  lua_setfield(L, -2, "name");

  // restore context
//  THCudaCheck(cudaSetDevice(oldDevice));

  return 1;
}
}

//static int cutorch_getState(lua_State *L)
//{
//  lua_getglobal(L, "cutorch");
//  lua_getfield(L, -1, "_state");
//  lua_remove(L, -2);
//  return 1;
//}

//const char *getDevicePropertiesName = "getDeviceProperties";

static const struct luaL_Reg clnn_stuff__ [] = {
//  {"synchronize", cutorch_synchronize},
//  {"getDevice", cutorch_getDevice},
//  {"deviceReset", cutorch_deviceReset},
//  {"getDeviceCount", cutorch_getDeviceCount},
  {"getDeviceProperties", clnn_getDeviceProperties},
//  {"getMemoryUsage", cutorch_getMemoryUsage},
//  {"setDevice", cutorch_setDevice},
//  {"seed", cutorch_seed},
//  {"seedAll", cutorch_seedAll},
//  {"initialSeed", cutorch_initialSeed},
//  {"manualSeed", cutorch_manualSeed},
//  {"manualSeedAll", cutorch_manualSeedAll},
//  {"getRNGState", cutorch_getRNGState},
//  {"setRNGState", cutorch_setRNGState},
//  {"getState", cutorch_getState},
  {NULL, NULL}
};

struct luaL_Reg makeReg( const char *name, lua_CFunction fn ) {
    struct luaL_Reg reg;
    reg.name = name;
    reg.func = fn;
    cout << "reg.name" << reg.name <<endl;
    return reg;
}

int luaopen_libclnn( lua_State *L ) {
  printf("luaopen_libclnn called :-)\n");
  cout << " try cout" << endl;

  lua_newtable(L);

//  struct luaL_Reg clnn_stuff[2];
 // cout << "clnn_getDeviceProperties" << (long)clnn_getDeviceProperties << endl;
 // clnn_stuff[0] = makeReg("getDeviceProperties", clnn_getDeviceProperties);
//  clnn_stuff[0] = makeReg("", 0);
 // clnn_stuff[1] = makeReg(NULL, NULL);

  luaL_setfuncs(L, clnn_stuff__, 0);
 // cout << "clnn_stuff[0]->name" << clnn_stuff[0].name << endl;
 // luaL_register(L, NULL, clnn_stuff);
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


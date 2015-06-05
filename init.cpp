#include <stdio.h>
#include <iostream>
#include "EasyCL.h"
using namespace std;

//#include "THClTensorRandom.h"

extern "C" {
  #include "lua.h"
  #include "utils.h"
  #include "luaT.h"
  #include "THClGeneral.h"
  int luaopen_libclnn( lua_State *L );
  extern void clnn_ClStorage_init(lua_State* L);
  extern void clnn_ClTensor_init(lua_State* L);
  //extern void clnn_ClTensorMath_init(lua_State* L);
  //extern void clnn_ClTensorOperator_init(lua_State* L);
}

namespace clnn {
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
  static int clnn_getDeviceCount(lua_State *L)
  {
    int count = easycl::DevicesInfo::getNumDevices();
    lua_pushnumber(L, count);
    return 1;
  }
  static int clnn_getDeviceProperties(lua_State *L)
  {
    cout << "clnn_getDeviceProperties" << endl;

    int device = (int)luaL_checknumber(L, 1)-1;
    cout << "device: " << device << endl;

    easycl::DeviceInfo deviceInfo = easycl::DevicesInfo::getDeviceInfo( device );
    lua_newtable(L);

    setProperty(L, "maxWorkGroupSize", deviceInfo.maxWorkGroupSize);
    setProperty(L, "platformVendor", deviceInfo.platformVendor);
    string deviceTypeString = "";
    if( deviceInfo.deviceType == 4 ) {
        deviceTypeString = "GPU";
    }
    if( deviceInfo.deviceType == 2 ) {
        deviceTypeString = "CPU";
    }
    if( deviceInfo.deviceType == 8 ) {
        deviceTypeString = "Accelerator";
    }
    setProperty(L, "deviceType", deviceTypeString);
    setProperty(L, "globalMemSizeMB", deviceInfo.globalMemSize / 1024 / 1024);
    setProperty(L, "localMemSizeKB", deviceInfo.localMemSize / 1024);
    setProperty(L, "globalMemCachelineSizeKB", deviceInfo.globalMemCachelineSize / 1024 );
    setProperty(L, "maxMemAllocSizeMB", deviceInfo.maxMemAllocSize / 1024 / 1024);
    setProperty(L, "maxComputeUnits", deviceInfo.maxComputeUnits);
    setProperty(L, "maxWorkGroupSize", deviceInfo.maxWorkGroupSize);
    setProperty(L, "deviceName", deviceInfo.deviceName);
    setProperty(L, "openClCVersion", deviceInfo.openClCVersion);
    setProperty(L, "deviceVersion", deviceInfo.deviceVersion);
    setProperty(L, "maxClockFrequency", deviceInfo.maxClockFrequency);

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
    {"getDeviceCount", clnn_getDeviceCount},
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

  clnn_ClStorage_init(L);
  clnn_ClTensor_init(L);
//  clnn_ClTensorMath_init(L);
//  clnn_ClTensorOperator_init(L);

  lua_pushlightuserdata(L, state);
  lua_setfield(L, -2, "_state");
  cout << "luaopen_libclnn done\n" << endl;

  return 1;
}


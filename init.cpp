#include <stdio.h>
#include <iostream>
#include "EasyCL.h"
using namespace std;

#include "util/StatefulTimer.h"

#include "commit_generated.h"

//#include "THClTensorRandom.h"

extern "C" {
  #include "lua.h"
  #include "utils.h"
  #include "luaT.h"
  int luaopen_libcltorch( lua_State *L );
  extern void cltorch_ClStorage_init(lua_State* L);
  extern void cltorch_ClTensor_init(lua_State* L);
  extern void cltorch_ClTensorMath_init(lua_State* L);
  extern void cltorch_ClTensorOperator_init(lua_State* L);
}

#include "THClGeneral.h"
#include "THClStorage.h"

namespace cltorch {
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
  static int cltorch_getDeviceCount(lua_State *L)
  {
    int count = easycl::DevicesInfo::getNumGpus();
    lua_pushnumber(L, count);
    return 1;
  }
  static int cltorch_getDevice(lua_State *L)
  {
    THClState *state = cltorch_getstate(L);
    int device = state->currentDevice;
    lua_pushnumber(L, device+1);
    return 1;
  }
  static int cltorch_setDevice(lua_State *L)
  {
    THClState *state = cltorch_getstate(L);
    int device = luaL_checknumber(L, 1) - 1;
    if(device < 0 || device >= state->allocatedDevices) {
       THError("Device doesnt exist");
    }
    state->currentDevice = device;
    return 0;
  }
  static int cltorch_synchronize(lua_State *L)
  {
    THClState *state = cltorch_getstate(L);
    THClState_getCl(state, state->currentDevice)->finish();
    return 0;
  }
  static int cltorch_getDeviceProperties(lua_State *L)
  {
    THClState *state = cltorch_getstate(L);
    int device = (int)luaL_checknumber(L, 1)-1;
    if(device < 0 || device >= state->allocatedDevices) {
       THError("Device doesnt exist");
    }

    easycl::DeviceInfo deviceInfo = easycl::DevicesInfo::getGpuInfo( device );
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

  static int cltorch_getState(lua_State *L)
  {
    lua_getglobal(L, "cltorch");
    lua_getfield(L, -1, "_state");
    lua_remove(L, -2);
    return 1;
  }

  static int cltorch_dumpTimings(lua_State *L)
  {
     StatefulTimer::timeCheck("before dump");
     StatefulTimer::dump( true );
     StatefulTimer::timeCheck("after dump");
//    lua_getglobal(L, "cltorch");
//    lua_getfield(L, -1, "_state");
//    lua_remove(L, -2);
    return 0;
  }

  // if you turn this to 1, you will see all copies of data between
  // host and gpu
  // useful for checking we're not doing this too often...
  static int cltorch_setTrace(lua_State *L)
  {
    THClState *state = cltorch_getstate(L);
    int trace = luaL_checknumber(L, 1);
    state->trace = trace;
//    THClStorage_traceOn = trace;
    return 0;
  }

  static int cltorch_setAddFinish(lua_State *L)
  {
    THClState *state = cltorch_getstate(L);
    int addFinish = luaL_checknumber(L, 1);
    state->addFinish = addFinish;
    return 0;
  }
  static int cltorch_about(lua_State *L)
  {
    cout << "cltorch.  OpenCL backend for Torch" << endl;
    cout << "Built from commit " << commit << endl;
    cout << "More info, doc: https://github.com/hughperkins/cltorch" << endl;
    cout << "Issues: https://github.com/hughperkins/cltorch/issues" << endl;
    return 0;
  }

  static const struct luaL_Reg cltorch_stuff__ [] = {
    {"getDevice", cltorch_getDevice},
    {"setDevice", cltorch_setDevice},
    {"synchronize", cltorch_synchronize},
    {"finish", cltorch_synchronize},
    {"getDeviceCount", cltorch_getDeviceCount},
    {"getDeviceProperties", cltorch_getDeviceProperties},
    {"getState", cltorch_getState},
    {"setTrace", cltorch_setTrace},
    {"setAddFinish", cltorch_setAddFinish},
    {"dumpTimings", cltorch_dumpTimings},
    {"about", cltorch_about},
    {NULL, NULL}
  };
}

int luaopen_libcltorch( lua_State *L ) {
  lua_newtable(L);
  luaL_setfuncs(L, cltorch::cltorch_stuff__, 0);

  THClState* state = (THClState*)malloc(sizeof(THClState));
  THClInit(state);

  cltorch_ClStorage_init(L);
  cltorch_ClTensor_init(L);
  cltorch_ClTensorMath_init(L);
  cltorch_ClTensorOperator_init(L);

  lua_pushlightuserdata(L, state);
  lua_setfield(L, -2, "_state");

  return 1;
}


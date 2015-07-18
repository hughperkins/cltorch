#include "UserKernel.h"

extern "C" {
  #include "lua.h"
  #include "utils.h"
  #include "luaT.h"
}
#include "EasyCL.h"
#include "THClKernels.h"

#include <iostream>
#include <string>
using namespace std;

class ClKernel {
public:
  int refCount;
  string source;
  string kernelName;
  string generatedSource;
  CLKernel *kernel;
};
//} ClKernel;

static int kernel(lua_State *L) {
  lua_pushstring(L, "hi there :-)");
//  UserKernel kernel;
  return 1;
}
//lua_State *L, const char *tname, const char *parenttname,
//                                       lua_CFunction constructor, lua_CFunction destructor, lua_CFunction factory);
static int printKernel(lua_State *L) {
  lua_pushstring(L, "hi there :-)");
  return 1;
}
static const struct luaL_Reg funcs[] = {
  {"kernel", kernel},
  {"printKernel", printKernel},
  {NULL, NULL}
};
static void ClKernel_rawInit(ClKernel *self) {
  self->refCount = 1;
  self->source = "";
}
static int ClKernel_new(lua_State *L) {
  cout << "ClKernel_new()" << endl;
  ClKernel *self = (ClKernel*)THAlloc(sizeof(ClKernel));
  self = new(self) ClKernel();
  ClKernel_rawInit(self);

  if(lua_type(L, 1) == LUA_TTABLE) {
    cout << "first param is a table" << endl;
    lua_getfield(L, 1, "src");
    const char*src_char = lua_tostring(L, -1);
    string source = src_char;
    cout << source << endl;
    self->source = source;
    THClState *state = cltorch_getstate(L);
    EasyCL *cl = THClState_getClv2(state, state->currentDevice);
    string kernelName = "user_kernel";  // can override by param, in future
    string generatedSource = "kernel void " + kernelName + "(";
    generatedSource += ") {\n";   // probalby should use ostringstream for this really, for speed
    generatedSource += source + "\n";
    generatedSource += "}\n";
    cout << "generatedSource: " << generatedSource << endl;
    self->generatedSource = generatedSource;
    self->kernelName = kernelName;
    self->kernel = cl->buildKernelFromString(generatedSource, kernelName, "", "user_kernel");
  } else {
    THError("First parameter to torch.ClKernel should be a table");
  }

  luaT_pushudata(L, self, "torch.ClKernel");
  cout << "ClKernel_new() finish" << endl;
  return 1;
}
static int ClKernel_free(lua_State *L) {
  cout << "ClKernel_free()" << endl;
  ClKernel *self = (ClKernel*)THAlloc(sizeof(ClKernel));
  if(!self) {
    return 0;
  }
  if(THAtomicDecrementRef(&self->refCount))
  {
    delete self->kernel;
    self->~ClKernel();
    THFree(self);
  }
  return 0;
}
static int ClKernel_factory(lua_State *L) {
  cout << "ClKernel_factory()" << endl;
  THError("not implemented");
  return 0;
}
static int ClKernel_print(lua_State *L) {
  cout << "ClKernel_print()" << endl;
  ClKernel *self = (ClKernel *)luaT_checkudata(L, 1, "torch.ClKernel");
  cout << "refCount=" << self->refCount << " source=" << self->source << endl;
  return 0;
}
static int ClKernel_run(lua_State *L) {
  cout << "ClKernel_run()" << endl;
  THClState *state = cltorch_getstate(L);
  ClKernel *self = (ClKernel *)luaT_checkudata(L, 1, "torch.ClKernel");
  cout << "refCount=" << self->refCount << " source=" << self->source << endl;
  self->kernel->run_1d(64, 64);  // obviously shoudl get these from somewhere, in future
  EasyCL *cl = THClState_getClv2(state, state->currentDevice);
//  cl->finish();  // not sure if we want this actually
  return 0;
}
static const struct luaL_Reg ClKernel_funcs [] = {
  {"print", ClKernel_print},
  {"run", ClKernel_run},
  {0,0}
};
void cltorch_UserKernel_init(lua_State *L)
{
  luaL_setfuncs(L, funcs, 0);

  luaT_newmetatable(L, "torch.ClKernel", NULL,
                    ClKernel_new, ClKernel_free, ClKernel_factory);
  luaL_setfuncs(L, ClKernel_funcs, 0);
  lua_pop(L, 1);
}


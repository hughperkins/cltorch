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
#include <vector>
using namespace std;

class ClKernelArg {
public:
  string name;
  ClKernelArg(string name) :
    name(name) {
  }
};

class ClKernelArgInt : public ClKernelArg {
public:
//  int value;
  ClKernelArgInt(string name) :
    ClKernelArg(name) {
//    value(value) {
  }
};

class ClKernelArgTensor : public ClKernelArg {
public:
//   value;
  ClKernelArgTensor(string name) :
    ClKernelArg(name) {
//  ClKernelArgTensor(int value) :
//    value(value) {
  }
};

class ClKernelArgFloat : public ClKernelArg {
public:
//  float value;
  ClKernelArgFloat(string name) :
    ClKernelArg(name) {
//  ClKernelArgFloat(float value) :
//    value(value) {
  }
};

class ClKernel {
public:
  int refCount;
  string source;
  string extraSource;
  string kernelName;
  string generatedSource;
  CLKernel *kernel;
  vector< ClKernelArg * >args;
  ClKernel() {
    kernel = 0;
    kernelName = "user_kernel";
    extraSource = "";
  }
  ~ClKernel() {
    for(int i = 0; i < (int)args.size(); i++) {
      delete args[i];
    }
    args.clear();
  }
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
static void printStack(string name, lua_State *L) {
  int top = lua_gettop(L);
  cout << name << " top: " << top << endl;
  for(int i = 1; i <= top; i++ ) {
    cout << "  stack[" << i << "] type=" << lua_typename(L, lua_type(L, i)) << " " << lua_type(L, i) << endl;
  }
}
static int ClKernel_new(lua_State *L) {
  cout << "ClKernel_new()" << endl;
  ClKernel *self = (ClKernel*)THAlloc(sizeof(ClKernel));
  self = new(self) ClKernel();
  ClKernel_rawInit(self);

  printStack("start", L);
  if(lua_type(L, 1) == LUA_TTABLE) {
//    cout << "first param is a table " << lua_typename(L, lua_type(L, 1)) << endl;
//    lua_getfield(L, 1, "src");
//    const char*src_char = lua_tostring(L, -1);
//    lua_pop(L, 1);
//    string source = src_char;
//    cout << source << endl;
//    self->source = source;

    // get args ...
//    cout << "gettop " << lua_gettop(L) << endl;
    cout << "gettop " << lua_gettop(L) << endl;
    printStack("before get input", L);
//    lua_getfield(L, 1, "input");
//    printStack("after get input", L);
//    cout << "after get input gettop " << lua_gettop(L) << endl;
//    cout << "is_table " << lua_istable(L, 1) << endl;
//    cout << "gettop, after pop " << lua_gettop(L) << endl;
//    cout << "gettop " << lua_gettop(L) << endl;
    lua_pushnil(L);
    while(lua_next(L, -2) != 0) {
//      printStack("after next", L);
//      cout << "  k,v=" << lua_tostring(L, -2) << " " << lua_tostring(L, -1) << endl;
      string key = lua_tostring(L, -2);
      cout << "key " << key << endl;
//      string value = lua_tostring(L, -1);  // could do a bit more validation here...
      if(key == "input") {
      } else if( key == "output") {
      } else if( key == "inout") {
      } else if( key == "name") {
        self->kernelName = lua_tostring(L, -1); // probably should use luaT for this
      } else if( key == "src") {
        self->source = lua_tostring(L, -1);
      } else if( key == "funcs") {
        self->extraSource = lua_tostring(L, -1);
      } else {
        THError("Parameter %s not recognized", key.c_str());
      }
      lua_pop(L, 1);
    }
    cout << "gettop " << lua_gettop(L) << endl;
    lua_pop(L, 1);
    cout << "gettop " << lua_gettop(L) << endl;

    // validate a bit
    if(self->source == "") {
      THError("Missing parameter src, or was empty");
    }

    THClState *state = cltorch_getstate(L);
    EasyCL *cl = THClState_getClv2(state, state->currentDevice);
    string kernelName = "user_kernel";  // can override by param, in future
    string generatedSource = "";
    generatedSource += self->extraSource + "\n";
    generatedSource += "kernel void " + kernelName + "(";
    generatedSource += ") {\n";   // probalby should use ostringstream for this really, for speed
    generatedSource += self->source + "\n";
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
//  THClState *state = cltorch_getstate(L);
  ClKernel *self = (ClKernel *)luaT_checkudata(L, 1, "torch.ClKernel");
  cout << "refCount=" << self->refCount << " source=" << self->source << endl;
  self->kernel->run_1d(64, 64);  // obviously shoudl get these from somewhere, in future
//  EasyCL *cl = THClState_getClv2(state, state->currentDevice);
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


#include "UserKernel.h"

extern "C" {
  #include "lua.h"
  #include "utils.h"
  #include "luaT.h"
}

#include <iostream>
#include <string>
using namespace std;

class ClKernel {
public:
  int refCount;
  string source;
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
    const char*src = lua_tostring(L, -1);
    cout << src << endl;
    self->source = src;
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
static const struct luaL_Reg ClKernel_funcs [] = {
  {"print", ClKernel_print},
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


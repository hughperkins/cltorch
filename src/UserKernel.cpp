#include "UserKernel.h"

extern "C" {
  #include "lua.h"
  #include "utils.h"
  #include "luaT.h"
}

#include <iostream>
#include <string>
using namespace std;

class UserKernel {
public:
  string source;
};
static int kernel(lua_State *L) {
  lua_pushstring(L, "hi there :-)");
  UserKernel kernel;
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
static int ClKernel_new(lua_State *L) {
  cout << "ClKernel_new()" << endl;
  return 0;
}
static int ClKernel_free(lua_State *L) {
  cout << "ClKernel_free()" << endl;
  return 0;
}
static int ClKernel_factory(lua_State *L) {
  cout << "ClKernel_factory()" << endl;
  return 0;
}
static int ClKernel_print(lua_State *L) {
  cout << "ClKernel_print()" << endl;
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


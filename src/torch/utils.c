#include "utils.h"

THLongStorage* cltorch_checklongargs(lua_State *L, int index)
{
  THLongStorage *storage;
  int i;
  int narg = lua_gettop(L)-index+1;

  if(narg == 1 && luaT_toudata(L, index, "torch.LongStorage"))
  {
    THLongStorage *storagesrc = luaT_toudata(L, index, "torch.LongStorage");
    storage = THLongStorage_newWithSize(storagesrc->size);
    THLongStorage_copy(storage, storagesrc);
  }
  else
  {
    storage = THLongStorage_newWithSize(narg);
    for(i = index; i < index+narg; i++)
    {
      if(!lua_isnumber(L, i))
      {
        THLongStorage_free(storage);
        luaL_argerror(L, i, "number expected");
      }
      THLongStorage_set(storage, i-index, lua_tonumber(L, i));
    }
  }
  return storage;
}

int cltorch_islongargs(lua_State *L, int index)
{
  int narg = lua_gettop(L)-index+1;

  if(narg == 1 && luaT_toudata(L, index, "torch.LongStorage"))
  {
    return 1;
  }
  else
  {
    int i;

    for(i = index; i < index+narg; i++)
    {
      if(!lua_isnumber(L, i))
        return 0;
    }
    return 1;
  }
  return 0;
}

struct THClState* cltorch_getstate(lua_State* L)
{
  lua_getglobal(L, "cltorch");
  lua_getfield(L, -1, "_state");
  struct THClState *state = lua_touserdata(L, -1);
  lua_pop(L, 2);
  return state;
}

#include <stdio.h>
#include "lua.h"

int luaopen_libclnn( lua_State *L ) {
    printf("luaopen_libclnn called :-)\n");
    return 1;
}


#include <stdio.h>
#include "lua.h"
#include <iostream>
using namespace std;

extern "C" {
    int luaopen_libclnn( lua_State *L );
}

int luaopen_libclnn( lua_State *L ) {
    printf("luaopen_libclnn called :-)\n");
    cout << " try cout" << endl;
    return 1;
}


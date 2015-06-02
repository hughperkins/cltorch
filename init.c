// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdio.h>
#include "lua.h"
#include <iostream>

int luaopen_libclnn( lua_State *L ) {
    printf("luaopen_libclnn called :-)\n");
    cout << " try cout" << endl;
    return 1;
}


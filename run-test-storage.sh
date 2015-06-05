#!/bin/bash

# Note: should be called from root directory, the one this script is in

#(cd build
#cmake ..  || exit 1
#make -j 4 || exit 1
#) || exit 1

source ~/torch/activate || exit 1
luarocks make rocks/clnn-scm-1.rockspec || exit 1
# LUA_CPATH="$LUA_CPATH;build/?.so" LUA_PATH="$LUA_PATH;?.lua" luajit test/test.lua

if [[ ! -v LUAEXE ]]; then {
    LUAEXE=luajit
} fi
echo using luaexe: ${LUAEXE}

if [[ x${RUNGDB} == x1 ]]; then {
  rungdb.sh ${LUAEXE} test/test-storage.lua
} else {
  ${LUAEXE} test/test-storage.lua
} fi


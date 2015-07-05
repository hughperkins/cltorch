#!/bin/bash

# Note: should be called from root directory, the one this script is in

if [[ -f ~/torch/activate ]]; then {
  source ~/torch/activate || exit 1
} fi
luarocks make rocks/cltorch-scm-1.rockspec || exit 1
# LUA_CPATH="$LUA_CPATH;build/?.so" LUA_PATH="$LUA_PATH;?.lua" luajit test/test.lua

export LUA_PATH="$LUA_PATH;${PWD}/thirdparty/?.lua"

if [[ ! -v LUAEXE ]]; then {
    LUAEXE=luajit
} fi
echo using luaexe: ${LUAEXE}

if [[ x${RUNGDB} == x1 ]]; then {
  rungdb.sh ${LUAEXE} test/cltorch-unit-tensor.lua $2 $3 $4 $5 $6 $7
} else {
  ${LUAEXE} test/cltorch-unit-tensor.lua $2 $3 $4 $5 $6 $7
} fi


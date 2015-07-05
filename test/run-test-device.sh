#!/bin/bash

# Note: should be called from root directory

source ~/torch/activate || exit 1
luarocks make rocks/cltorch-scm-1.rockspec || exit 1

if [[ ! -v LUAEXE ]]; then {
    LUAEXE=luajit
} fi
echo using luaexe: ${LUAEXE}

if [[ x${RUNGDB} == x1 ]]; then {
  rungdb.sh ${LUAEXE} test/test-device.lua
} else {
  ${LUAEXE} test/test-device.lua
} fi


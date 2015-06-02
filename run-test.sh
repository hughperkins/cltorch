#!/bin/bash

# Note: should be called from `build` subdirectory

cmake .. || exit 1
make -j 4 || exit 1

(cd ..
source ~/torch/activate
source ~/git/luarocks/activate
LUA_CPATH="$LUA_CPATH;build/?.so" LUA_PATH="$LUA_PATH;?.lua" luajit test/test.lua
)


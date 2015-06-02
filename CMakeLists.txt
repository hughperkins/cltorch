cmake_minimum_required( VERSION 2.8.12 )

Find_package(Lua51 REQUIRED)
include_directories(${LUA_INCLUDE_DIR})

add_library( clnn SHARED init.cpp )



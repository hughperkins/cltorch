cmake_minimum_required( VERSION 2.8.12 )

Find_package(Lua51 REQUIRED)
include_directories(${LUA_INCLUDE_DIR})

FIND_PACKAGE(Torch REQUIRED)

add_library( clnn SHARED init.cpp )

ADD_SUBDIRECTORY(lib)

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/lib/THC")


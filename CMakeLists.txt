cmake_minimum_required( VERSION 2.8.12 )

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

#Find_package(Lua51 REQUIRED)
#include_directories(${LUA_INCLUDE_DIR})

FIND_PACKAGE(Torch REQUIRED)

# add_library( clnn SHARED init.cpp )

SET(CMAKE_C_FLAGS "-std=c99 -Werror=implicit-function-declaration")
SET(CMAKE_CXX_FLAGS "-std=c++0x -Wall")

ADD_SUBDIRECTORY(lib)

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/lib/THCl")

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/torch")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/EasyCL")

# SET(src Storage.c init.c Tensor.c TensorMath.c TensorOperator.c torch/utils.c)
SET(src init.cpp torch/utils.c)
SET(luasrc init.lua test/test.lua)

ADD_TORCH_PACKAGE(clnn "${src}" "${luasrc}")
TARGET_LINK_LIBRARIES(clnn luaT THCL)


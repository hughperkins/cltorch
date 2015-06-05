cmake_minimum_required( VERSION 2.8.12 )

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

#Find_package(Lua51 REQUIRED)
#include_directories(${LUA_INCLUDE_DIR})

FIND_PACKAGE(Torch REQUIRED)

# add_library( clnn SHARED init.cpp )

SET(CMAKE_C_FLAGS "-std=c99 -Werror=implicit-function-declaration -save-temps")
SET(CMAKE_CXX_FLAGS "-std=c++0x -Wall -save-temps")


ADD_SUBDIRECTORY(lib)

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/lib/THCl")

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/torch")
# INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/DeepCL")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/DeepCL/EasyCL")

# SET(src Storage.c init.c Tensor.c TensorMath.c TensorOperator.c torch/utils.c)
SET(src init.cpp torch/utils.c Storage.c)
SET(luasrc init.lua test/test.lua)

ADD_TORCH_PACKAGE(clnn "${src}" "${luasrc}")

set( EasyCL_sources EasyCL/EasyCL.cpp
    EasyCL/deviceinfo_helper.cpp EasyCL/platforminfo_helper.cpp EasyCL/CLWrapper.cpp
    EasyCL/DevicesInfo.cpp EasyCL/DeviceInfo.cpp
    EasyCL/CLKernel.cpp EasyCL/thirdparty/clew/src/clew.c )
#add_library( EasyCL SHARED ${EasyCL_sources} )
#link_libraries( EasyCL )
#INSTALL(TARGETS EasyCL
#          RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}"
#          LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
#          ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}")

# INCLUDE(DeepCL)

# message("deepcl src ${deepcl_sources}")

ADD_SUBDIRECTORY( DeepCL EXCLUDE_FROM_ALL )

# ADD_LIBRARY( DeepCL SHARED 

TARGET_LINK_LIBRARIES(clnn luaT THCL EasyCL DeepCL)


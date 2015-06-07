cmake_minimum_required( VERSION 2.8.12 )

OPTION(DEV_RUN_COG "clnn maintainers only, otherwise set to 'OFF'." OFF)

# remove unused options
set(LIBPNGPP_AVAILABLE FALSE CACHE BOOL "libjpeg available" FORCE)
mark_as_advanced(FORCE LIBPNGPP_AVAILABLE)
set(LIBJPEG_AVAILABLE FALSE CACHE BOOL "libjpeg available" FORCE)
mark_as_advanced(FORCE LIBJPEG_AVAILABLE)
set(BUILD_LUA_WRAPPERS FALSE CACHE BOOL "libjpeg available" FORCE)
mark_as_advanced(FORCE BUILD_LUA_WRAPPERS)

if( "${CMAKE_SYSTEM_NAME}" STREQUAL "Linux" )
  set( ON_LINUX 1 )
endif()
if( "${CMAKE_SYSTEM_NAME}" STREQUAL "Windows" )
  set( ON_WINDOWS 1 )
endif()

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

#Find_package(Lua51 REQUIRED)
#include_directories(${LUA_INCLUDE_DIR})

FIND_PACKAGE(Torch REQUIRED)

# add_library( clnn SHARED init.cpp )

SET(CMAKE_C_FLAGS "-std=c99 -Werror=implicit-function-declaration -save-temps")
SET(CMAKE_CXX_FLAGS "-std=c++0x -Wall")


ADD_SUBDIRECTORY(lib)

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/lib/THCl")

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/torch")
# INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/DeepCL")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/DeepCL/EasyCL")

# SET(src Storage.c init.c Tensor.c TensorMath.c TensorOperator.c torch/utils.c)
SET(src init.cpp torch/utils.c Storage.c Tensor.c TensorMath.c)
SET(luasrc init.lua Tensor.lua )

ADD_TORCH_WRAP(clnntensormathwrap TensorMath.lua)

ADD_TORCH_PACKAGE(clnn "${src}" "${luasrc}")

if( ON_LINUX ) 
    execute_process( COMMAND git --version RESULT_VARIABLE git_test ERROR_QUIET )
    if( ${git_test} EQUAL 0 )
        set( GIT_FOUND 1 )
    endif()
    if( GIT_FOUND )
        add_custom_target( 
            git_clnn_rec
            git submodule update --init --force --recursive
        )
        add_dependencies( clnn git_clnn_rec )
    endif()
endif()

#set( EasyCL_sources EasyCL/EasyCL.cpp
#    EasyCL/deviceinfo_helper.cpp EasyCL/platforminfo_helper.cpp EasyCL/CLWrapper.cpp
#    EasyCL/DevicesInfo.cpp EasyCL/DeviceInfo.cpp
#    EasyCL/CLKernel.cpp EasyCL/thirdparty/clew/src/clew.c )
#add_library( EasyCL SHARED ${EasyCL_sources} )
#link_libraries( EasyCL )
#INSTALL(TARGETS EasyCL
#          RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}"
#          LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
#          ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}")

# INCLUDE(DeepCL)

# message("deepcl src ${deepcl_sources}")

#set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)
ADD_SUBDIRECTORY( DeepCL )

# ADD_LIBRARY( DeepCL SHARED 
TARGET_LINK_LIBRARIES(clnn luaT THCL EasyCL DeepCL)



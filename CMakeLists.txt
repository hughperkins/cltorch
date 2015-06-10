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

#SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

FIND_PACKAGE(Torch REQUIRED)

SET(CMAKE_C_FLAGS "-std=c99 -Werror=implicit-function-declaration -save-temps")
SET(CMAKE_CXX_FLAGS "-std=c++0x -Wall")


SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

INCLUDE("${CMAKE_MODULE_PATH}/build_clBLAS.cmake")
INCLUDE_DIRECTORIES(${CLBLAS_INCLUDE_DIRS})
LINK_DIRECTORIES(${CLBLAS_LIBRARY_DIR})

INCLUDE("${CMAKE_MODULE_PATH}/build_DeepCL.cmake")
INCLUDE_DIRECTORIES(${DEEPCL_INCLUDE_DIRS})
LINK_DIRECTORIES(${DEEPCL_LIBRARY_DIR})

ADD_SUBDIRECTORY(lib)

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/lib/THCl")

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/torch")
# INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/DeepCL")
#INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/DeepCL/EasyCL")

SET(src init.cpp torch/utils.c Storage.c Tensor.c TensorMath.c
  TensorOperator.c)
SET(luasrc init.lua Tensor.lua )

ADD_TORCH_WRAP(clnntensormathwrap TensorMath.lua)

ADD_TORCH_PACKAGE(clnn "${src}" "${luasrc}")
ADD_DEPENDENCIES(clnn clBLAS)
ADD_DEPENDENCIES(clnn DeepCL)

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

# ADD_SUBDIRECTORY( DeepCL )

TARGET_LINK_LIBRARIES(clnn luaT THCL )
TARGET_LINK_LIBRARIES( clnn ${DEEPCL_LIBARIES} )
TARGET_LINK_LIBRARIES( clnn ${CLBLAS_LIBRARIES} )



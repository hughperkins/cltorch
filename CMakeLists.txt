cmake_minimum_required( VERSION 2.8.12 )

OPTION(DEV_RUN_COG "cltorch maintainers only, otherwise set to 'OFF'." OFF)

if( "${CMAKE_SYSTEM_NAME}" STREQUAL "Linux" )
  set( ON_LINUX 1 )
endif()
if( "${CMAKE_SYSTEM_NAME}" STREQUAL "Windows" )
  set( ON_WINDOWS 1 )
endif()

#SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

FIND_PACKAGE(Torch REQUIRED)

SET(CMAKE_C_FLAGS "-std=c99 -Werror=implicit-function-declaration")
SET(CMAKE_CXX_FLAGS "-std=c++0x -Wall")


SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

INCLUDE("${CMAKE_MODULE_PATH}/build_clBLAS.cmake")
INCLUDE_DIRECTORIES(${CLBLAS_INCLUDE_DIRS})
LINK_DIRECTORIES(${CLBLAS_LIBRARY_DIR})

INCLUDE("${CMAKE_MODULE_PATH}/build_EasyCL.cmake")
INCLUDE_DIRECTORIES(${EASYCL_INCLUDE_DIRS})
LINK_DIRECTORIES(${EASYCL_LIBRARY_DIR})

ADD_SUBDIRECTORY(lib)

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/lib/THCl")

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/torch")

SET(src init.cpp torch/utils.c Storage.c Tensor.c TensorMath.c
  TensorOperator.c)
SET(luasrc init.lua Tensor.lua Random.lua )

ADD_TORCH_WRAP(cltorchtensormathwrap TensorMath.lua)

ADD_TORCH_PACKAGE(cltorch "${src}" "${luasrc}")
ADD_DEPENDENCIES(cltorch clBLAS)
ADD_DEPENDENCIES(cltorch EasyCL)
ADD_DEPENDENCIES(cltorchtensormathwrap EasyCL)

if( ON_LINUX ) 
    execute_process( COMMAND git --version RESULT_VARIABLE git_test ERROR_QUIET )
    if( ${git_test} EQUAL 0 )
        set( GIT_FOUND 1 )
    endif()
    if( GIT_FOUND )
        add_custom_target( 
            git_cltorch_rec
            git submodule update --init --force --recursive
        )
        add_dependencies( cltorch git_cltorch_rec )
    endif()
endif()

TARGET_LINK_LIBRARIES(cltorch luaT THCl )
TARGET_LINK_LIBRARIES( cltorch ${EASYCL_LIBARIES} )
TARGET_LINK_LIBRARIES( cltorch ${CLBLAS_LIBRARIES} )


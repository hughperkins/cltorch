cmake_minimum_required(VERSION 2.8.12)

OPTION(DEV_RUN_COG "cltorch maintainers only, otherwise set to 'OFF'." OFF)

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  set(ON_LINUX 1)
endif()
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
  set(ON_WINDOWS 1)
endif()

#SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

FIND_PACKAGE(Torch REQUIRED)

SET(CMAKE_C_FLAGS "-std=c99 -Werror=implicit-function-declaration")
SET(CMAKE_CXX_FLAGS "-std=c++0x -Wall")


SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/cmake")

execute_process(COMMAND bash -c "echo $PWD")
execute_process(COMMAND bash -c "echo $PWD"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
execute_process(COMMAND git status
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    execute_process(
      COMMAND git submodule update --init --force --recursive
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      )

INCLUDE("${CMAKE_MODULE_PATH}/build_clBLAS.cmake")
INCLUDE_DIRECTORIES(${CLBLAS_INCLUDE_DIRS})
LINK_DIRECTORIES(${CLBLAS_LIBRARY_DIR})

INCLUDE("${CMAKE_MODULE_PATH}/build_EasyCL.cmake")
INCLUDE_DIRECTORIES(${EASYCL_INCLUDE_DIRS})
LINK_DIRECTORIES(${EASYCL_LIBRARY_DIR})

ADD_SUBDIRECTORY(src)

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/src")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/src/lib")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/src/torch")

SET(src src/UserKernel.cpp src/init.cpp src/torch/utils.c src/Storage.cpp src/Tensor.cpp TensorMath.c
  src/TensorOperator.c)
SET(luasrc src/init.lua src/FFI.lua src/Tensor.lua src/Random.lua src/Test.lua src/test/unit_storage.lua src/test/unit_tensor.lua)

ADD_TORCH_WRAP(cltorchtensormathwrap src/TensorMath.lua)

ADD_TORCH_PACKAGE(cltorch "${src}" "${luasrc}")
ADD_DEPENDENCIES(cltorch clBLAS)
ADD_DEPENDENCIES(cltorch EasyCL)
ADD_DEPENDENCIES(cltorch EasyCL-external)
ADD_DEPENDENCIES(cltorchtensormathwrap EasyCL)
#add_custom_target(

execute_process(COMMAND git log -n 1 --pretty=%h OUTPUT_VARIABLE git_commit OUTPUT_STRIP_TRAILING_WHITESPACE)
#execute_process(COMMAND echo string commit="${git_commit}" > ${CMAKE_CURRENT_SOURCE_DIR}/commit.h)
file(GENERATE OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/cltorch_commit_generated.h 
  CONTENT "const char *cltorch_commit=\"${git_commit}\";\n"
)

if(DEV_RUN_COG)
    add_custom_target(
        cog_cltorch
        python ${CMAKE_CURRENT_SOURCE_DIR}/src/EasyCL/thirdparty/cogapp/cog.py -q -I ${CMAKE_CURRENT_SOURCE_DIR}/src/EasyCL/cog-batteries -r ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_dependencies(cltorch cog_cltorch)
endif(DEV_RUN_COG)


#if(ON_LINUX) 
    execute_process(COMMAND git --version RESULT_VARIABLE git_test ERROR_QUIET)
    if(${git_test} EQUAL 0)
        set(GIT_FOUND 1)
    endif()
    if(GIT_FOUND)
        add_custom_target(
            git_cltorch_rec
            git submodule update --init --force --recursive
            COMMAND sed -i -e "s/-pedantic//" ${CMAKE_CURRENT_SOURCE_DIR}/src/clMathLibraries/clBLAS/src/CMakeLists.txt
            COMMAND sed -i -e "s/-Wall -Wextra/-w/" ${CMAKE_CURRENT_SOURCE_DIR}/src/clMathLibraries/clBLAS/src/CMakeLists.txt
            COMMAND sed -i -e "s/-Wstrict-prototypes/-w/" ${CMAKE_CURRENT_SOURCE_DIR}/src/clMathLibraries/clBLAS/src/CMakeLists.txt
            COMMAND sed -i -e "s/${CMAKE_CXX_FLAGS}/${CMAKE_CXX_FLAGS} -w /" ${CMAKE_CURRENT_SOURCE_DIR}/src/clMathLibraries/clBLAS/src/CMakeLists.txt
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
        add_dependencies(cltorch git_cltorch_rec)
        add_dependencies(clBLAS-external git_cltorch_rec)
        add_dependencies(EasyCL-external git_cltorch_rec)
    endif()
#endif()

TARGET_LINK_LIBRARIES(cltorch luaT THCl)
TARGET_LINK_LIBRARIES(cltorch ${EASYCL_LIBARIES})
TARGET_LINK_LIBRARIES(cltorch ${CLBLAS_LIBRARIES})


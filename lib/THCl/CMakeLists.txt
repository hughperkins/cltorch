cmake_minimum_required(VERSION 2.8)

OPTION(DEV_RUN_COG "cltorch maintainers only, otherwise set to 'OFF'." OFF)

SET(CMAKE_C_FLAGS "-std=c99")
#SET(src
#    THCGeneral.c THCAllocator.c THCStorage.c THCStorageCopy.c THCTensor.c THCTensorCopy.c)
SET(src
    THClGeneral.cpp THClStorage.cpp THClStorageCopy.cpp THClTensor.cpp THClTensorCopy.cpp THClTensorMath.cpp 
    THClTensorMathPointwise.cpp THClReduceApplyUtils.cpp THClApply.cpp
    THClTensorMathCompare.cpp THClTensorMathCompareT.cpp
    THClTensorMathPairwise.cpp THClTensorMath2.cpp
    THClBlas.cpp THClTensorMathBlas.cpp THClBlas.cpp )
set(src-cl)

message("CLBLAS_INCLUDE_DIRS ${CLBLAS_INCLUDE_DIRS}")
INCLUDE_DIRECTORIES(${CLBLAS_INCLUDE_DIRS})

ADD_LIBRARY(THCL SHARED ${src} ${src-cl})
TARGET_LINK_LIBRARIES(THCL TH )
message("DEEPCL_LIBRARIES ${EASYCL_LIBRARIES}")
TARGET_LINK_LIBRARIES( THCL ${EASYCL_LIBRARIES} )
TARGET_LINK_LIBRARIES( THCL ${CLBLAS_LIBRARIES} )
add_dependencies( THCL clBLAS )
add_dependencies( THCL EasyCL )

if(DEV_RUN_COG)
    add_custom_target(
        cog_thcl
        python ${CMAKE_CURRENT_SOURCE_DIR}/../../DeepCL/thirdparty/cogapp/cog.py -q -I ${CMAKE_CURRENT_SOURCE_DIR}/../../DeepCL/cog-batteries -r ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_dependencies( THCL cog_thcl )
endif(DEV_RUN_COG)

INSTALL(TARGETS THCL
          RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}"
          LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
          ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}")


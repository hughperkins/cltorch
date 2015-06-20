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
    THClBlas.cpp THClTensorMathBlas.cpp THClBlas.cpp THClReduce.cpp
    THClTypeParseTraits.cpp THClReduceAll.cpp THClDeviceUtils.cpp
    THClTensorMasked.cpp THClTensorMathTransformReduce.cpp
    THClTensorIndex.cpp THClKernels.cpp )
set(src-cl)

message("CLBLAS_INCLUDE_DIRS ${CLBLAS_INCLUDE_DIRS}")
INCLUDE_DIRECTORIES(${CLBLAS_INCLUDE_DIRS})

ADD_LIBRARY(THCl SHARED ${src} ${src-cl})
TARGET_LINK_LIBRARIES(THCl TH )
message("DEEPCL_LIBRARIES ${EASYCL_LIBRARIES}")
TARGET_LINK_LIBRARIES( THCl ${EASYCL_LIBRARIES} )
TARGET_LINK_LIBRARIES( THCl ${CLBLAS_LIBRARIES} )
add_dependencies( THCl clBLAS )
add_dependencies( THCl EasyCL )

if(DEV_RUN_COG)
    add_custom_target(
        cog_thcl
        python ${CMAKE_CURRENT_SOURCE_DIR}/../../EasyCL/thirdparty/cogapp/cog.py -q -I ${CMAKE_CURRENT_SOURCE_DIR}/../../EasyCL/cog-batteries -r ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_dependencies( THCl cog_thcl )
endif(DEV_RUN_COG)

INSTALL(FILES
          THCl.h
          THClGeneral.h
          THClBlas.h
          THClStorage.h
	  THClStorageCopy.h
          THClTensor.h
	  THClTensorCopy.h
#          THClTensorRandom.h
          THClTensorMath.h
#          THClTensorConv.h
#          THClTensorSort.h
          THClApply.h
          THClReduce.h
          THClReduceApplyUtils.h
#	  THClAllocator.h
          DESTINATION "${Torch_INSTALL_INCLUDE_SUBDIR}/THCl")

INSTALL(TARGETS THCl
          RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}"
          LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
          ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}")


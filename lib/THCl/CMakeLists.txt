SET(CMAKE_C_FLAGS "-std=c99")
#SET(src
#    THCGeneral.c THCAllocator.c THCStorage.c THCStorageCopy.c THCTensor.c THCTensorCopy.c)
SET(src
    THClGeneral.cpp THClStorage.cpp THClStorageCopy.cpp THClStorageCopy.c )
set(src-cl)
# add_library( THCl SHARED THClStorage.c THClGeneral.c )

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../DeepCL//EasyCL)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../DeepCL/src)

ADD_LIBRARY(THCL SHARED ${src} ${src-cl})
TARGET_LINK_LIBRARIES(THCL TH )

INSTALL(TARGETS THCL
          RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}"
          LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
          ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}")


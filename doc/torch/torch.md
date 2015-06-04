# torch

## root /

* init.c
  * contains luaopen_libtorch
  * creates `torch` table
  * calls torch_(type)(Storage|Tensor)_init
* init.lua
  * requires libtorch
  * includes (Tensor|File|FFI|...).lua
  * defines torch.(type|class|include|...)
* Storage.c
  * torch_Storage_(NAME) => torch_(Type)Storage_(NAME)
  * torch_Storage => torch.(Type)Storage
  * includes generic/Storage.c and THGenerateAllTypes.h
* Tensor.c
  * includes generic/Tensor.c and THGenerateAllTypes.h
* Tensor.lua
  * A few Tensor utility methods, like print, expand, totable
  * Tensor.(typeAs|byte|char|short|int|long|float|double) methods

## /generic

* Storage.c
  * handles the lua types/interfaces, then calls THStorage methods
  * for char,byte,int,double,float, ...:
      * torch_(Type)Storage_new 
         * calls TH(Type)Storage_new , with appropriate size
         * calls TH(Type)Storage_set, with values
      * torch_(Type)Storage_free
         * calls TH(Type)Storage_free
      * torch_(Type)Storage_resize
         * calls TH(Type)Storage_resize
         * (does little else)
      * torch_(Type)Storage_copy
         * if statement, baseed on source type, passed in as argument
         * calls appropriate THStorage_copy, based on source type
      * torch_(Type)Storage_fill
         * calls TH(Type)Storage_fill
      * torch_(Type)Storage_newindx
         * calls TH(Type)Storage_set
      * torch_(Type)Storage_index
         * calls TH(Type)Storage_get
      * torch_(Type)Storage_factory
         * calls TH(Type)Storage_new
      * torch_(type)Storage_init
         * registers the methods above
* Tensor.c
  * for char,byte,int,double,float, ...:
      * torch_(Type)Tensor_new 
         * creates a THLongStorage to read size of each dimension
         * creates a new TH(Type)Storage, and resizes it
         * calls TH(Type)Storage_set on each item

## lib/TH

* TH.h
  * includes lib/TH/TH*.h
* THGeneral.c/h
  * THAlloc
  * THFree
  * THRealloc
  * THSetArgErrorHandler
  * (a few others)
* THStorage.h
  * THStorage => TH(Type)Storage
  * THStorage_(Name) => TH(Type)Storage_(Name)
  * includes lib/TH/generic/THStorage.h and THGenerateAllTypes.h
  * includes lib/TH/generic/THStorageCopy.h and THGenerateAllTypes.h
* THStorage.c
  * includes lib/TH/generic/THStorage.c and all types
  * includes lib/TH/generic/THStorageCopy.c and all types
* THTensor.c/h
  * generates all types for lib/TH/generic/THTensor*.h/c

## lib/TH/generic

* THStorage.c/h
  * TH(Type)Storage_(new,newWithSize,newWithAllocator,free,newWithData,
    resize, fill, set, get)
* THTensor.c/h
  * TH(Type)Tensor_(storage,storageOffset,size,stride,data,rawInit,
       new,newWithTensor,newWithStorage,newWithSize,newClone, resize, ...)

# cutorch

## root /

* init.lua
  * require libcutorch
  * include Tensor.lua, FFI.lua, test,la
* init.c
  * defines and registers global functions, eg synchronize, getNumStreams, setDevice
  * calls cutorch_Cuda(Storage|Tensor|TensorMath|TensorOperator)_init(L)
  * intializes THCState, and stores it as _state
* Storage.c
  * calls generic/Storage.c for Real=Cuda
  * defines cutorch_CudaStorage_copy for src type (Cuda|Byte|...)
  * defines cutorch_(Type)Storage_copy for all src types
  * registers the copy methods as 'copy' method of torch.ByteStorage etc

  * seems like since generic/Storage.c just calls appropriate THCuda method, that generic/Storage.c doesnt need much modificatoin?
* Tensor.c
  * as for Storage.c: include generic/Tensor.c for every type, just overwite `copy` methods
* Tensor.lua
  * injects a `cuda()` method to each of the other Tensor types
  * adds 'double()', 'float()' etc method to torch.CudaTensor type
* FFI.lua
  * almost empty
  * contains the structs, ie:
    * THCState
    * THCudaStorage
    * THCudaTensor
  * adds `cdata` and `data` methods (?) to Storage and Tensor

## torch/generic

* Storage.c and Tensor.c from torch/generic, -no change-, but modified some... eg THCState instead of THState, and cutorch_getState, instead of checkudata and not the only difference :-(

## lib/THC

* THC.h
  * includes lib/THC/TH*.h
* THCGeneral.h/c
  * includes cuda.h etc
  * defines THAssert
  * defines THC_API, THC_EXTERNC
  * defines THCState struct
  * implementation for global functions, like:
    * THCudaInit
    * THCudaBlas_init
    * THCState_getNumDevices
* THCStorage.c/h/cu
  * defines THCudaStorage struct, containing allocator, refcount, ..
  * defines THCudaStorage_(new,set,get,free,fill,resize,data)
  * `fill` and `resize` are in .cu, presumably because these need kernels (seems like .cu is just more definitions of what are in the .h file though, some in the .c, some in the .cu)
  * other methods just use cudaMalloc, cudaFree, cudaMemcpy, etc
* THCTensor.c/h/cu
  * various methods like retain, free, set1d/2d/..., get1d/2d/...
    squeeze, storage, new, data, lots of `new` methods, `resize` methods
  * meld shows there's basically no difference between the .c file and the original torch one, in lib/TH/generic/THTensor.c
  * the cu has two functions:
    * THCudaTensor_getDevice
    * THCudaTensor_getTextureObject
* (No lib/THC/generic)


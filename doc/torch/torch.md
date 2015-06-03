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

* init.c
* init.lua
* Storage.c
* Tensor.c
* Tensor.lua

## torch/generic

* Storage.c and Tensor.c from torch/generic

## lib/THC

* THC.h
* THCStorage.c/h
* THCStorage.cu
* THCTensor.c/h
* THCTensor.cu
* (No lib/THC/generic)


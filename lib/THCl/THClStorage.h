#ifndef THCL_STORAGE_INC
#define THCL_STORAGE_INC

#include "THStorage.h"
#include "THClGeneral.h"

#define TH_STORAGE_REFCOUNTED 1
#define TH_STORAGE_RESIZABLE  2
#define TH_STORAGE_FREEMEM    4

extern int THClStorage_traceOn;

typedef struct THClStorage
{
  int device;
  float *data; // I know this seems a bit superfluous....
  struct EasyCL *cl;
  struct CLWrapper *wrapper;
  long size;
  int refcount;
  char flag;
  THAllocator *allocator;
  void *allocatorContext;
  struct THClStorage *view;
} THClStorage;


THCL_API float* THClStorage_data(THClState *state, const THClStorage*);
THCL_API long THClStorage_size(THClState *state, const THClStorage*);

/* slow access -- checks everything */
//THCL_API void THClStorage_set(THClState *state, THClStorage*, long, float);
//THCL_API float THClStorage_get(THClState *state, const THClStorage*, long);

THCL_API THClStorage* THClStorage_new(THClState *state);
THCL_API THClStorage* THClStorage_newWithSize(THClState *state, long size);
THCL_API THClStorage* THClStorage_newWithSize1(THClState *state, float);
THCL_API THClStorage* THClStorage_newWithSize2(THClState *state, float, float);
THCL_API THClStorage* THClStorage_newWithSize3(THClState *state, float, float, float);
THCL_API THClStorage* THClStorage_newWithSize4(THClState *state, float, float, float, float);
THCL_API THClStorage* THClStorage_newWithMapping(THClState *state, const char *filename, long size, int shared);

/* takes ownership of data */
THCL_API THClStorage* THClStorage_newWithData(THClState *state, float *data, long size);

THCL_API THClStorage* THClStorage_newWithAllocator(THClState *state, long size,
                                                      THAllocator* allocator,
                                                      void *allocatorContext);
THCL_API THClStorage* THClStorage_newWithDataAndAllocator(
    THClState *state, float* data, long size, THAllocator* allocator, void *allocatorContext);

THCL_API void THClStorage_setFlag(THClState *state, THClStorage *storage, const char flag);
THCL_API void THClStorage_clearFlag(THClState *state, THClStorage *storage, const char flag);
THCL_API void THClStorage_retain(THClState *state, THClStorage *storage);

THCL_API void THClStorage_free(THClState *state, THClStorage *storage);
THCL_API void THClStorage_resize(THClState *state, THClStorage *storage, long size);
THCL_API void THClStorage_fill(THClState *state, THClStorage *storage, float value);

#endif

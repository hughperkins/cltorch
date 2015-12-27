local ok, ffi = pcall(require, 'ffi')
if ok then

   local cdefs = [[

typedef struct THClStorage
{
  int device;
  float *data;
  void *cl;
  void *wrapper;
  long size;
  int refcount;
  char flag;
  void *allocator;
  void *allocatorContext;
  struct THClStorage *view;
} THClStorage;

typedef struct THClTensor
{
    long *size;
    long *stride;
    int nDimension;

    THClStorage *storage;
    long storageOffset;
    int refcount;

    char flag;

    int device;
} THClTensor;

]]
   ffi.cdef(cdefs)

   local Storage = torch.getmetatable('torch.ClStorage')
   local Storage_tt = ffi.typeof('THClStorage**')

   rawset(Storage, "cdata", function(self) return Storage_tt(self)[0] end)
   rawset(Storage, "data", function(self) return Storage_tt(self)[0].data end)
   -- Tensor
   local Tensor = torch.getmetatable('torch.ClTensor')
   local Tensor_tt = ffi.typeof('THClTensor**')

   rawset(Tensor, "cdata", function(self) return Tensor_tt(self)[0] end)

   rawset(Tensor, "data",
          function(self)
             self = Tensor_tt(self)[0]
             return self.storage ~= nil and self.storage.data + self.storageOffset or nil
          end
   )

end

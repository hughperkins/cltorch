#ifndef THCL_GENERAL_INC
#define THCL_GENERAL_INC

#include "THGeneral.h"
#include "THAllocator.h"
#undef log1p

#ifdef __cplusplus
# define THCL_EXTERNC extern "C"
#else
# define THCL_EXTERNC extern
#endif

#ifdef WIN32
# ifdef THCL_EXPORTS
#  define THCL_API THCL_EXTERNC __declspec(dllexport)
# else
#  define THCL_API THCL_EXTERNC __declspec(dllimport)
# endif
#else
# define THCL_API THCL_EXTERNC
#endif

//// from http://stackoverflow.com/questions/295120/c-mark-as-deprecated
//#ifdef __GNUC__
//#define DEPRECATED __attribute__((deprecated))
//#elif defined(_MSC_VER)
//#define DEPRECATED __declspec(deprecated)
//#else
//#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
//#define DEPRECATED
//#endif

#ifdef __GNUC__
#define DEPRECATED_POST __attribute__((deprecated))
#endif

struct EasyCL;
struct CLWrapper;
struct DeviceInfo;

#ifdef __cplusplus
#include <iostream>
#endif // __cplusplus

typedef struct THClScratchSpace {
  struct CLWrapper *wrapper;
  float *data;
} THClScratchSpace;

/* Global state to be held in the cltorch table. */
typedef struct THClState
{
  int allocatedDevices;
  int currentDevice;
  int trace; // default 0; set to 1 to see message for every gpu buffer alloc, delete,
             // or device <-> host transfer
  int addFinish; // default 0, should we add clFinish() after any kernel, enqueue, etc?
                 // (good for debugging stuff, bad for perf)
  int detailedTimings;
  struct THClScratchSpace**scratchSpaceByDevice; // for now, do one 'stream' per device
                                 // can improve later...
  struct DeviceInfo **deviceInfoByDevice;
//  int *workgroupSizeByDevice;
  struct EasyCL **clByDevice;
 // EasyCL *getCl();  
} THClState;

THCL_API void THClInit(THClState* state);
THCL_API void THClShutdown(THClState* state);
//THCL_API void THClEnablePeerToPeerAccess(THClState* state);

/* State manipulators and accessors */
THCL_API int THClState_getNumDevices(THClState* state);
THCL_API void THClState_setDevice(THClState* state, int device);
THCL_API int THClState_getDevice(THClState* state);
THCL_API struct EasyCL *THClState_getCl(THClState* state) DEPRECATED_POST;
THCL_API struct EasyCL *THClState_getClAndDevice(THClState* state, int *p_device) DEPRECATED_POST;
THCL_API struct EasyCL *THClState_getClv2(THClState* state, int device);

//THCL_API void THClState_reserveStreams(THClState* state, int numStreams);
//THCL_API int THClState_getNumStreams(THClState* state);

//THCL_API cudaStream_t THClState_getDeviceStream(THClState *state, int device, int stream);
//THCL_API cudaStream_t THClState_getCurrentStream(THClState *state);
//THCL_API int THClState_getCurrentStreamIndex(THClState *state);
//THCL_API void THClState_setStream(THClState *state, int device, int stream);
//THCL_API void THClState_setStreamForCurrentDevice(THClState *state, int stream);

//THCL_API void THClState_reserveBlasHandles(THClState* state, int numHandles);
//THCL_API int THClState_getNumBlasHandles(THClState* state);

//THCL_API clblasHandle_t THClState_getDeviceBlasHandle(THClState *state, int device, int handle);
//THCL_API clblasHandle_t THClState_getCurrentBlasHandle(THClState *state);
//THCL_API int THClState_getCurrentBlasHandleIndex(THClState *state);
//THCL_API void THClState_setBlasHandle(THClState *state, int device, int handle);
//THCL_API void THClState_setBlasHandleForCurrentDevice(THClState *state, int handle);

/* For the current device and stream, returns the allocated scratch space */
THCL_API struct THClScratchSpace* THClState_getCurrentDeviceScratchSpace(THClState* state) DEPRECATED_POST;
THCL_API struct THClScratchSpace* THClState_getDeviceScratchSpace(THClState* state, int device, int stream);
THCL_API size_t THClState_getCurrentDeviceScratchSpaceSize(THClState* state) DEPRECATED_POST;
THCL_API size_t THClState_getDeviceScratchSpaceSize(THClState* state, int device);

//#define THClCheck(err)  __THClCheck(err, __FILE__, __LINE__)
//#define THCublasCheck(err)  __THCublasCheck(err,  __FILE__, __LINE__)

//THCL_API void __THClCheck(cudaError_t err, const char *file, const int line);
//THCL_API void __THCublasCheck(clblasStatus_t status, const char *file, const int line);

// define dim3, since this came from cuda in cutorch
#ifdef __cplusplus
class dim3 {
public:
    uint vec[3];
    size_t vec_for_cl[3];
//    size_t vec_size_t[3];
    dim3() {
        vec[0] = 1;
        vec[1] = 1;
        vec[2] = 1;
    }
    dim3( uint32_t x ) {
        vec[0] = x;
        vec[1] = 1;
        vec[2] = 1;
    }
    dim3( uint32_t x, uint32_t y ) {
        vec[0] = x;
        vec[1] = y;
        vec[2] = 1;
    }
    dim3( uint32_t x, uint32_t y, uint32_t z ) {
        vec[0] = x;
        vec[1] = y;
        vec[2] = z;
    }
    inline uint32_t x() {
        return vec[0];
    }
    inline uint32_t y() {
        return vec[1];
    }
    inline uint32_t z() {
        return vec[2];
    }
    size_t const *as_size_t() {
        for( int i = 0; i < 3; i++ ) {
            vec_for_cl[i] = vec[i];
        }
        return vec_for_cl;
    }
};

std::ostream &operator<<( std::ostream &os, const dim3 &obj );

//typedef struct _dim3 {
//    int x;
//    int y;
//    int z;
//    _dim3( int x ) {
//        this->x = x;
//        y = 1;
//        z = 1;
//    }
//} dim3;
#endif // __cplusplus

// seems that min is really inconsistent across standard libraires, lets just make our own ... :-/
inline int mymin( int a, int b ) {
    return a < b ? a : b;
}

#endif


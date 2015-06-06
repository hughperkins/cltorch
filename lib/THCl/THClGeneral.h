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

//#define THClCheck(err)  __THClCheck(err, __FILE__, __LINE__)
//THCL_API void __THClCheck(clError_t err, const char *file, const int line);

struct EasyCL;

/* Global state to be held in the cutorch table. */
typedef struct THClState
{
  struct EasyCL *cl;
} THClState;

THCL_API void THClInit(THClState* state);
THCL_API void THClShutdown(THClState* state);


// define dim3, since this came from cuda in cutorch
#ifdef __cplusplus
class dim3 {
public:
    int x;
    int y;
    int z;
    dim3() {
        x = 1;
        y = 1;
        z = 1;
    }
    dim3( int x ) {
        this->x = x;
        y = 1;
        z = 1;
    }
    dim3( int x, int y ) {
        this->x = x;
        this->y = y;
        z = 1;
    }
    dim3( int x, int y, int z ) {
        this->x = x;
        this->y = y;
        this->z = z;
    }
};
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


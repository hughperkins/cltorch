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

/* Global state to be held in the cutorch table. */
typedef struct THClState
{
} THClState;

THCL_API void THClInit(THClState* state);
THCL_API void THClShutdown(THClState* state);
#endif


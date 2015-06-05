extern "C" {
    #include "THClGeneral.h"
    #include "TH.h"
}

#include <stdio.h>
#include "EasyCL.h"

//#include "THCTensorRandom.h"
//#include "THCBlas.h"
//#include "THCAllocator.h"

void THClInit(THClState* state)
{
    printf("*******************************************\n");
    printf("THClInit()\n");
  state->cl = EasyCL::createForFirstGpuOtherwiseCpu(); // obviously this should change...
}

void THClShutdown(THClState* state)
{
  delete state->cl;
    printf("THClShutdown()\n");
    printf("*******************************************\n");
}


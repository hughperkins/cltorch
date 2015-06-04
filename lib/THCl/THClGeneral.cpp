#include "THClGeneral.h"
#include "TH.h"
#include <stdio.h>
#include "EasyCL.h"

//#include "THCTensorRandom.h"
//#include "THCBlas.h"
//#include "THCAllocator.h"

void THClInit(THClState* state)
{
    printf("THClInit()\n");
  state->cl = new EasyCL();
}

void THClShutdown(THClState* state)
{
    printf("THClShutdown()\n");
  delete state->cl;
}


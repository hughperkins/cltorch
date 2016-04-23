#pragma once

#include "THClTensor.h"

void THClDebug_printTensor(THClState *state, THClTensor *target);
void THClDebug_printSize(THClState *state, THClTensor *target);
void THClDebug_printSize(const char *name, THClState *state, THClTensor *target);


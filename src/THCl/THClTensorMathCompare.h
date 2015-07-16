#pragma once

void THClTensor_logicalValue(THClState *state, THClTensor *self_, THClTensor *src, HasOperator2 *op);
void THClTensor_logicalTensor(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2, HasOperator3 *op);


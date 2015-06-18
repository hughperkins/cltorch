#ifndef TH_CL_TENSOR_MATH_INC
#define TH_CL_TENSOR_MATH_INC

#include "THClTensor.h"
#include "THClGeneral.h"

THCL_API void THClTensor_fill(THClState *state, THClTensor *self, float value);
THCL_API void THClTensor_zero(THClState *state, THClTensor *self);

THCL_API void THClTensor_zeros(THClState *state, THClTensor *r_, THLongStorage *size);
THCL_API void THClTensor_ones(THClState *state, THClTensor *r_, THLongStorage *size);
THCL_API void THClTensor_reshape(THClState *state, THClTensor *r_, THClTensor *t, THLongStorage *size);
THCL_API long THClTensor_numel(THClState *state, THClTensor *t);

THCL_API void THClTensor_add(THClState *state, THClTensor *self, THClTensor *src, float value);
THCL_API void THClTensor_mul(THClState *state, THClTensor *self, THClTensor *src, float value);
THCL_API void THClTensor_div(THClState *state, THClTensor *self, THClTensor *src, float value);


THCL_API void THClTensor_cadd(THClState *state, THClTensor *self, THClTensor *src1, float value, THClTensor *src2);
THCL_API void THClTensor_cmul(THClState *state, THClTensor *self, THClTensor *src1, THClTensor *src2);
THCL_API void THClTensor_cpow(THClState *state, THClTensor *self, THClTensor *src1, THClTensor *src2);
THCL_API void THClTensor_cdiv(THClState *state, THClTensor *self, THClTensor *src1, THClTensor *src2);

THCL_API void THClTensor_addcmul(THClState *state, THClTensor *self, THClTensor* t, float value, THClTensor *src1, THClTensor *src2);
THCL_API void THClTensor_addcdiv(THClState *state, THClTensor *self, THClTensor* t, float value, THClTensor *src1, THClTensor *src2);

THCL_API float THClTensor_dot(THClState *state, THClTensor *self, THClTensor *src);

THCL_API float THClTensor_minall(THClState *state, THClTensor *self);
THCL_API float THClTensor_maxall(THClState *state, THClTensor *self);
THCL_API float THClTensor_sumall(THClState *state, THClTensor *self);
THCL_API float THClTensor_prodall(THClState *state, THClTensor *self);
//THCL_API void THClTensor_min(THClState *state, THClTensor *values, THClTensor *indices, THClTensor *src, long dim);
//THCL_API void THClTensor_max(THClState *state, THClTensor *values, THClTensor *indices, THClTensor *src, long dim);
THCL_API void THClTensor_min(THClState *state, THClTensor *self, THClTensor *src, long dim);
THCL_API void THClTensor_max(THClState *state, THClTensor *self, THClTensor *src, long dim);
THCL_API void THClTensor_sum(THClState *state, THClTensor *self, THClTensor *src, long dim);
THCL_API void THClTensor_prod(THClState *state, THClTensor *self, THClTensor *src, long dim);
THCL_API void THClTensor_cumsum(THClState *state, THClTensor *self, THClTensor *src, long dim);
THCL_API void THClTensor_cumprod(THClState *state, THClTensor *self, THClTensor *src, long dim);

THCL_API void THClTensor_addmv(THClState *state, THClTensor *self, float beta, THClTensor *t, float alpha, THClTensor *mat, THClTensor *vec);
THCL_API void THClTensor_addmm(THClState *state, THClTensor *self, float beta, THClTensor *t, float alpha, THClTensor *mat1, THClTensor *mat2);
THCL_API void THClTensor_addr(THClState *state, THClTensor *self, float beta, THClTensor *t, float alpha, THClTensor *vec1, THClTensor *vec2);
THCL_API void THClTensor_baddbmm(THClState *state, THClTensor *result, float beta, THClTensor *t,
                                  float alpha, THClTensor *batch1, THClTensor *batch2);

THCL_API void THClTensor_log(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_log1p(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_exp(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_cos(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_acos(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_cosh(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_sin(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_asin(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_sinh(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_tan(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_atan(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_tanh(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_pow(THClState *state, THClTensor *self, THClTensor *src, float value);
THCL_API void THClTensor_tpow(THClState *state, THClTensor *self, float value, THClTensor *src);
THCL_API void THClTensor_clamp(THClState *state, THClTensor *self, THClTensor *src, float min_value, float max_value);
THCL_API void THClTensor_sqrt(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_ceil(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_floor(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_abs(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_sign(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_round(THClState *state, THClTensor *self, THClTensor *src);
TH_API void THClTensor_atan2(THClState *state, THClTensor *r_, THClTensor *tx, THClTensor *ty);

// this doesnt exist in torch or cutorch. original development :-P
THCL_API void THClTensor_neg(THClState *state, THClTensor *self, THClTensor *src);
THCL_API void THClTensor_sub(THClState *state, THClTensor *self, THClTensor *src, float value);
THCL_API void THClTensor_csub(THClState *state, THClTensor *self, THClTensor *src1, float value, THClTensor *src2);



THCL_API void THClTensor_ltValue(THClState *state, THClTensor *self_, THClTensor *src, float value);
THCL_API void THClTensor_gtValue(THClState *state, THClTensor *self_, THClTensor *src, float value);
THCL_API void THClTensor_leValue(THClState *state, THClTensor *self_, THClTensor *src, float value);
THCL_API void THClTensor_geValue(THClState *state, THClTensor *self_, THClTensor *src, float value);
THCL_API void THClTensor_eqValue(THClState *state, THClTensor *self_, THClTensor *src, float value);
THCL_API void THClTensor_neValue(THClState *state, THClTensor *self_, THClTensor *src, float value);

THCL_API void THClTensor_ltTensor(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2);
THCL_API void THClTensor_gtTensor(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2);
THCL_API void THClTensor_leTensor(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2);
THCL_API void THClTensor_geTensor(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2);
THCL_API void THClTensor_eqTensor(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2);
THCL_API void THClTensor_neTensor(THClState *state, THClTensor *self_, THClTensor *src1, THClTensor *src2);

THCL_API float THClTensor_meanall(THClState *state, THClTensor *self);
THCL_API void  THClTensor_mean(THClState *state, THClTensor *self, THClTensor *src, long dim);
THCL_API float THClTensor_varall(THClState *state, THClTensor *self);
THCL_API void  THClTensor_var(THClState *state, THClTensor *self, THClTensor *src, long dim, int flag);
THCL_API float THClTensor_stdall(THClState *state, THClTensor *self);
THCL_API void  THClTensor_std(THClState *state, THClTensor *self, THClTensor *src, long dim, int flag);
THCL_API float THClTensor_normall(THClState *state, THClTensor *self, float value);
THCL_API void  THClTensor_norm(THClState *state, THClTensor* self, THClTensor* src, float value, long dimension);
THCL_API void  THClTensor_renorm(THClState *state, THClTensor* self, THClTensor* src, float value, long dimension, float max_norm);
THCL_API float THClTensor_dist(THClState *state, THClTensor *self, THClTensor *src, float value);

THCL_API void THClTensor_rand(THClState *state, THClTensor *r_, THLongStorage *size);
THCL_API void THClTensor_randn(THClState *state, THClTensor *r_, THLongStorage *size);

THCL_API void THClTensor_indexCopy(THClState *state, THClTensor *res_, int dim, THLongTensor *indices, THClTensor *src);
THCL_API void THClTensor_indexFill(THClState *state, THClTensor *tensor, int dim, THLongTensor *index, float val);
THCL_API void THClTensor_indexSelect(THClState *state, THClTensor *tensor, THClTensor *src, int dim, THLongTensor *index);

THCL_API void THClTensor_maskedFill(THClState *state, THClTensor *tensor, THClTensor *mask, float value);
THCL_API void THClTensor_maskedCopy(THClState *state, THClTensor *tensor, THClTensor *mask, THClTensor *src);
THCL_API void THClTensor_maskedSelect(THClState *state, THClTensor *tensor, THClTensor *src, THClTensor *mask);

THCL_API void THClTensor_maskedFillByte(THClState *state, THClTensor *tensor, THByteTensor *mask, float value);
THCL_API void THClTensor_maskedCopyByte(THClState *state, THClTensor *tensor, THByteTensor *mask, THClTensor *src);
THCL_API void THClTensor_maskedSelectByte(THClState *state, THClTensor *tensor, THClTensor *src, THByteTensor *mask);

THCL_API int THClTensor_logicalall(THClState *state, THClTensor *self);
THCL_API int THClTensor_logicalany(THClState *state, THClTensor *self);

#endif


#include "torch/utils.h"
#include "luaT.h"
#include "THCl.h"

static int clnn_ClTensorOperator___add__(lua_State *L)
{
  THClTensor *tensor1 = luaT_toudata(L, 1, "torch.ClTensor");
  THClTensor *tensor2 = luaT_toudata(L, 2, "torch.ClTensor");
  THClTensor *r;
  THClState *state = clnn_getstate(L);
  THAssert(THClTensor_checkGPU(state, 2, tensor1, tensor2));

  if(!tensor1 && !tensor2)
    luaL_error(L, "expecting two Tensors or one Tensor and one number");
  else
  {
    r = THClTensor_new(state);
    luaT_pushudata(L, r, "torch.ClTensor");

    if(!tensor1 && tensor2)
    {
      THClTensor_resizeAs(state, r, tensor2);
      THClTensor_copy(state, r, tensor2);
      THClTensor_add(state, r, r, luaL_checknumber(L, 1));
    }
    else if(tensor1 && !tensor2)
    {
      THClTensor_resizeAs(state, r, tensor1);
      THClTensor_copy(state, r, tensor1);
      THClTensor_add(state, r, r, luaL_checknumber(L, 2));
    }
    else
    {
      THClTensor_resizeAs(state, r, tensor1);
      THClTensor_copy(state, r, tensor1);
      THClTensor_cadd(state, r, r, 1, tensor2);
    }
  }
  return 1;
}

static int clnn_ClTensorOperator___sub__(lua_State *L)
{
  THClTensor *tensor1 = luaT_toudata(L, 1, "torch.ClTensor");
  THClTensor *tensor2 = luaT_toudata(L, 2, "torch.ClTensor");
  THClTensor *r;
  THClState *state = clnn_getstate(L);
  THAssert(THClTensor_checkGPU(state, 2, tensor1, tensor2));

  if(!tensor1 && !tensor2)
    luaL_error(L, "expecting two Tensors or one Tensor and one number");
  else
  {
    r = THClTensor_new(state);
    luaT_pushudata(L, r, "torch.ClTensor");

    if(!tensor1 && tensor2)
    {
      THClTensor_resizeAs(state, r, tensor2);
      THClTensor_fill(state, r, luaL_checknumber(L, 1));
      THClTensor_cadd(state, r, r, -1, tensor2);
    }
    else if(tensor1 && !tensor2)
    {
      THClTensor_resizeAs(state, r, tensor1);
      THClTensor_copy(state, r, tensor1);
      THClTensor_add(state, r, r, -luaL_checknumber(L, 2));
    }
    else
    {
      THClTensor_resizeAs(state, r, tensor1);
      THClTensor_copy(state, r, tensor1);
      THClTensor_cadd(state, r, r, -1, tensor2);
    }
  }
  return 1;
}

static int clnn_ClTensorOperator___unm__(lua_State *L)
{
  THClTensor *tensor = luaT_checkudata(L, 1, "torch.ClTensor");
  THClTensor *r;
  THClState *state = clnn_getstate(L);
  THAssert(THClTensor_checkGPU(state, 1, tensor));

  r = THClTensor_new(state);
  luaT_pushudata(L, r, "torch.ClTensor");
  THClTensor_resizeAs(state, r, tensor);
  THClTensor_copy(state, r, tensor);
  THClTensor_mul(state, r, r, -1);

  return 1;
}

static int clnn_ClTensorOperator___mul__(lua_State *L)
{
  THClTensor *tensor1 = luaT_toudata(L, 1, "torch.ClTensor");
  THClTensor *tensor2 = luaT_toudata(L, 2, "torch.ClTensor");
  THClTensor *r;
  THClState *state = clnn_getstate(L);
  THAssert(THClTensor_checkGPU(state, 2, tensor1, tensor2));

  if(!tensor1 && !tensor2)
    luaL_error(L, "expecting two Tensors or one Tensor and one number");
  else
  {
    r = THClTensor_new(state);
    luaT_pushudata(L, r, "torch.ClTensor");

    if(!tensor1 && tensor2)
    {
      THClTensor_resizeAs(state, r, tensor2);
      THClTensor_copy(state, r, tensor2);
      THClTensor_mul(state, r, r, luaL_checknumber(L, 1));
    }
    else if(tensor1 && !tensor2)
    {
      THClTensor_resizeAs(state, r, tensor1);
      THClTensor_copy(state, r, tensor1);
      THClTensor_mul(state, r, r, luaL_checknumber(L, 2));
    }
    else
    {
      int dimt = tensor1->nDimension;
      int dims = tensor2->nDimension;

      if(dimt == 1 && dims == 1)
        lua_pushnumber(L, THClTensor_dot(state, tensor1, tensor2)); /* ok, we wasted r, but who cares */
      else if(dimt == 2 && dims == 1)
      {
        THClTensor_resize1d(state, r, tensor1->size[0]);
        THClTensor_zero(state, r);
        THClTensor_addmv(state, r, 1, r, 1, tensor1, tensor2);
      }
      else if(dimt == 2 && dims == 2)
      {
        THClTensor_resize2d(state, r, tensor1->size[0], tensor2->size[1]);
        THClTensor_zero(state, r);
        THClTensor_addmm(state, r, 1, r, 1, tensor1, tensor2);
      }
      else
        luaL_error(L, "multiplication between %dD and %dD tensors not yet supported", tensor1->nDimension, tensor2->nDimension);
    }
  }
  return 1;
}

static int clnn_ClTensorOperator___div__(lua_State *L)
{
  THClTensor *tensor = luaT_checkudata(L, 1, "torch.ClTensor");
  THClTensor *r;
  THClState *state = clnn_getstate(L);
  THAssert(THClTensor_checkGPU(state, 1, tensor));

  luaL_argcheck(L, lua_isnumber(L,2), 2, "number expected");

  r = THClTensor_new(state);
  luaT_pushudata(L, r, "torch.ClTensor");

  THClTensor_resizeAs(state, r, tensor);
  THClTensor_copy(state, r, tensor);
  THClTensor_mul(state, r, r, 1/lua_tonumber(L, 2));

  return 1;
}

static const struct luaL_Reg clnn_ClTensorOperator__ [] = {
  {"__add__", clnn_ClTensorOperator___add__},
  {"__sub__", clnn_ClTensorOperator___sub__},
  {"__unm__", clnn_ClTensorOperator___unm__},
  {"__mul__", clnn_ClTensorOperator___mul__},
  {"__div__", clnn_ClTensorOperator___div__},
  {NULL, NULL}
};

void clnn_ClTensorOperator_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.ClTensor");
  luaL_setfuncs(L, clnn_ClTensorOperator__, 0);
  lua_pop(L, 1);
}

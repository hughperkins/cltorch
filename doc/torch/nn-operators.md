# nn operators

This documents which maths operators are used by key nn modules, such as Linear, dropout, max pooling, and convolution.  In fact, those modules, though not necessarily those names

`t`, `t1`, `t2` means `Tensor` object

## Linear

* setting weights randomly (can just do from cpu probably)
* t:resize(size)
* t:copy(t2)
* t:add(t1, t2)
* t:addmv(1, t1, t2)
* t:addmm(0, t1, 1, t2, t3)
* t:addr(1, t1, t2)

* c methods:
  * maybe none? there is no generic/Linear.c file

## SpatialDropout

* t:resizeAs(t1)
* t:copy(t1)
* t:resize(t1, 1, 1)
* t:cmul(t1)
* t:mul(0.5f)

* c methods?

## SpatialMaxPooling

* bunch of c implementation in generic/SpatialMaxPooling

## SpatialZeroPadding

I guess we wouldnt want to actually really pad things, but might be easy option initially perhaps, should check what cutorch/cunn does perhaps?

* copy
* size
* narrow
* dim()
* zero()

* c methods?

## Tanh

* nothing in Tanh.lua
* in generic/Tanh.c:
  * bunch of stuff...

## SpatialConvolution

* bunch of c implementation in generic/SpatialConvolution.c

## Relu

* not much in lua
* some implementation in generic/PReLU.c possibly?

## DotProduct

* dot
* copy
* mul
* resizeAs
* no c implementation

## SoftMax

* not much in lua
* c implementation in generic/SoftMax.c



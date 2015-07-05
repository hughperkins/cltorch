# cltorch

An OpenCL backend for [torch](http://torch.ch/).

## What's working

### Import

<table>

<tr><td>Component<td>Status<td>Examples of what works now</tr>

<tr><td><pre>require 'cltorch'</pre> <td> works <td><pre>require 'cltorch'</pre></tr>

</table>

### Device information and control

<table>

<tr><td>Component<td>Status<td>Examples of what works now</tr>

<tr><td>Device information<td>works<td><pre>
print('num devices: ', cltorch.getDeviceCount())
props = cltorch.getDeviceProperties(1)
for k,v in props do
    print(k, v)
end
cltorch.setDevice(1)
print('current device: ', cltorch.getDevice())
cltorch.synchronize()
cltorch.finish() -- alias for synchronize()
</pre></tr>
</table>

### From [storage.md](https://github.com/torch/torch7/blob/master/doc/storage.md)

<table>

<tr><td>Component<td>Status<td>Examples of what works now</tr>

<tr><td> torch.ClStorage <td> works <td><pre>
c = torch.ClStorage()
c = torch.ClStorage(3)
c = torch.ClStorage{4,9,2}
c:fill(7)
a = torch.Storage{1.5, 2.4, 5.3}
c:copy(a)
a:copy(c)
d = torch.ClStorage(3)
d:copy(c)
print('size:', #d)
print('size:', d:size())
c:resize(5)
</pre></tr>

</table>

### From [tensor.md](https://github.com/torch/torch7/blob/master/doc/tensor.md)

<table>

<tr><td>Component<td>Status<td>Examples of what works now</tr>

<tr><td>torch.ClTensor constructors<td>Done<td><pre>
c = torch.ClTensor()
c = torch.ClTensor{7,4,5}
c = torch.ClTensor(3,2)
</pre></tr>

<tr><td>Cloning <td>Done<td><pre>
c = torch.ClTensor{{3,1,6},{2.1,5.2,3.9}}
d = c:clone()
print('isTensor', torch.isTensor(c))
</pre></tr>

<tr><td>Quering size and structure<td>Done<td><pre>
c = torch.ClTensor{3,5,2}
print('torch.isTensor(c)', torch.isTensor(c))
print('c:nDimension()', c:nDimension())
C = torch.ClTensor{{3,2,4},{9,7,5}}
print('C:nDimension()', C:nDimension())
print('c:dim()', C:dim())
print('C:size()', C:size())
print('C:size(1)', C:size(1))
print('C:size(2)', C:size(2))
print('#C', #C)
print('C:stride()', C:stride())
print('C:stride(1)', C:stride(1))
print('C:stride(2)', C:stride(2))
print('C:storage()', C:storage())
print('C:nElement()', C:nElement())
print('C:storageOffset()', C:storageOffset())
</pre></tr>

<tr><td>Querying elements<td>Removed<td><pre>
-- whilst it's technically trivial to create a kernel to read/write
-- a single element, for any more than one element, it is far more
-- efficient to just copy either the whole tensor to/from main memory,
-- or that particular part of the tensor you want to read
-- so, for now I've disabled querying single elements
-- If you a. have a use-case for read-writing single elements
--        b. can think of a way of not causing this to be accidentally
--           abused, please raise an issue
</pre></tr>

<tr><td>Copying and initializing<td>Done<td><pre>
c = torch.Tensor{2,6,9}:cl()
b = c:float()
c = torch.ClTensor{{3,1,6},{2.1,5.2,3.9}}
b:copy(c)
c:copy(b)
d = torch.ClTensor(2,3)
d:copy(c)
c:fill(1.345)
c:zero()
</pre></tr>

<tr><td>Resizing<td>Done<td><pre>
c:resize(3,2)
l = torch.LongStorage{3,3}
c:resize(l)
d = torch.ClTensor(2,2)
d:resizeAs(c)
</pre></tr>

<tr><td>Extracting sub-tensors<td>70%<td><pre>
E:narrow(1,2,3)
E:sub(2,3,2,2)
E:select(1,2):fill(99)
x[{ 2,{2,4} }] = 2 
x[{ {},4 }] = -1
x[{ {},2 }] = torch.range(1,5) 
x[torch.lt(x,0)] = -2
C:indexFill(2, torch.LongTensor{1,3}, -12)
x:indexCopy(2,torch.LongTensor{5,1},z)
x:indexSelect( ... )
x:maskedFill( ... )
x:gather( ... )
x:scatter( ... )
</pre></tr>


<tr><td>Expanding/Replicating/Squeezing Tensors<td>Done<td><pre>
torch.expand(x,3,2)
torch.expandAs(x, a)
result:repeatTensor(a, 3,2)
squeezed = a:squeeze()
</pre></tr>

<tr><td>Manipulating the tensor view<td>Done<td><pre>
C = torch.ClTensor{{3,2,4},{9,7,5}}
print(C:t())
print(C:transpose(1,2))
</pre></tr>

</table>

### From [random.md](https://github.com/torch/torch7/blob/master/doc/random.md)

<table>

<tr><td>Component<td>Status<td>Examples of what works now</tr>

<tr><td>uniform, etc <td>2%<td><pre>For now, you would need to do for example:
torch.Tensor(5,3):uniform():cl()
torch.Tensor(5,3):bernoulli() -- works now, but basically generates on host side
     -- then copies to gpu
</pre></tr>
</table>

### From [serialization.md](https://github.com/torch/torch7/blob/master/doc/serialization.md)

<table>

<tr><td>Component<td>Status<td>Examples of what works now</tr>

<tr><td>Serialization <td>50%<td><pre>
torch.save('filename.dat', torch.ClTensor{3,5,2})
a = torch.load('filename.dat')
</pre></tr>
</table>

### From [maths.md](https://github.com/torch/torch7/blob/master/doc/maths.md)

<table>

<tr><td>Component<td>Status<td>Examples of what works now</tr>


<tr><td>Construction or extraction functions<td>0%<td><pre>
-- For now, you can create using a FloatTensor,
-- then use the :cl operator to convert to a ClTensor, eg:
a = torch.Tensor(40,30):uniform():cl()
b = torch.eye(20):cl()
-- etc ...
</tre></tr>

<tr><td>Element-wise operations<td>Done<td><pre>
c:abs()
for _,name in ipairs({'log','exp', 'cos', 'acos', 'sin',
   'asin', 'atan', 'tanh', 'ceil', 'floor', 'abs', 
   'round'}) do
  loadstring('c:' .. name .. '()')()
end
</pre>
</tr>

<tr><td>basic operations <td>50% done<td><pre>
d = torch.ClTensor{{3,5,-2},{2.1,2.2,3.9}}
c = torch.ClTensor{{4,2,-1},{3.1,1.2,4.9}}
c:add(d)
c:cmul(d)
c:cdiv(d)
c:add(3)
c:mul(3)
c:div(2)
c = torch.add(c,3)
c = torch.mul(c, 4)
c = torch.div(c, 3)
torch.pow(2,c)
c:pow(2)
torch.cpow(c,d)
torch.cdiv(c,d)
torch.pow(c,2)
torch.clamp(c, 50, 100)
c:clamp(50, 100)
-c

A = torch.ClTensor{{1,2,-1},
                   {3,4,0}}
B = torch.ClTensor{{0,1},
                   {1,2},
                   {4,5}}
print(torch.mm(A,B))
C:mm(A,B)

C = torch.ClTensor{{3,2},{9,7}}
D = torch.ClTensor{{3,1,7},{3,2,4}}
E = torch.ClTensor{{3,1},{2,9},{3,2}}
print(torch.addmm(C,D,E))

c = torch.ClTensor{3,2}
D = torch.ClTensor{{3,1,7},{3,2,4}}
e = torch.ClTensor{3,1,2}
print(torch.addmv(c,D,e))

v1 = torch.ClTensor{3,5,1}
v2 = torch.ClTensor{2,4,8}
print(torch.dot(v1, v2))

print(torch.mv(A,v1))

C = torch.ClTensor{{3,1,7},{3,2,4},{8,5,3}}
d = torch.ClTensor{3,2,5}
e = torch.ClTensor{3,1,2}
print(torch.addr(C,d,e))
</pre></tr>

<tr><td>Logical operations <td>Done<td><pre>
d = torch.ClTensor{{3,5,-2},{2.1,2.2,3.9}}
c = torch.ClTensor{{4,2,-1},{3.1,1.2,4.9}}
for _,name in ipairs({'lt','le','gt','ge','ne','eq'}) do
  print(loadstring('return c:' .. name .. '(5)')())
end
for _,name in ipairs({'lt','le','gt','ge','ne','eq'}) do
  print(loadstring('return torch.' .. name .. '(c,d)')())
end
torch.any(torch.Tensor({0,1,0}):cl())
torch.all(torch.Tensor({0,1,0}):cl())
</pre></tr>

<tr><td>Overloaded operators <td>80% done<td><pre>
d = torch.ClTensor{{3,5,-2},{2.1,2.2,3.9}}
c = torch.ClTensor{{4,2,-1},{3.1,1.2,4.9}}
c = c + d
c = c - d
c = c / 2
c = c * 1.5
c = c + 4
c = c - 5

A = torch.ClTensor{{1,2,-1},
                   {3,4,0}}
B = torch.ClTensor{{0,1},
                   {1,2},
                   {4,5}}
print( A * B)

v1 = torch.ClTensor{3,5,1}
v2 = torch.ClTensor{2,4,8}
print(v1 * v2)
</pre></tr>

<tr><td>Column or row-wise operations<td>30% done<td><pre>
A = torch.ClTensor{{3,5,2},{4,5,6}}
A:sum()
A:sum(2)
A:sum(1)
print(torch.prod(A))
print(A:prod())
print(A:prod(1))
print(A:prod(2))
A:max()
result, idx = A:max(1)
result, idx = A:max(2)
A:min()
result, idx = A:min(1)
result, idx = A:min(2)
torch.cumsum(x, 1)
torch.cumsum(x, 2)
torch.cumprod(x, 1)
torch.cumprod(x, 2)
</pre></tr>

<tr><td>Matrix-wide operations<td>25%<td><pre>
A = torch.ClTensor{{3,5,2},{4,5,6}}
torch.norm(A)
torch.norm(A,1)
torch.norm(A,0)
torch.numel(A)
</pre></tr>

<tr><td>Original, not in torch or cutorch<td>N/A<td><pre>
c:csub(d) -- subtracts d from c, element-wise
         -- similar to 'c - d'
         -- but stores results into c
a:neg() -- similar to '- a'
        -- but stores results into a
c:apply("x = sqrt(x + 3.5)")
        -- note: a string, not a lua function
        -- this will be passed to OpenCL kernel :-)
-- Update: in fact, this is available for cutorch :-)
-- It is here: https://github.com/szagoruyko/cutorch-rtc
c:map(d, "x = 1000 * x + y * 10")
        -- note: a string, not a lua function
        -- this will be passed to OpenCL kernel :-)
c:map2(d, e, "x = sqrt(1000 * x + y * 10 + z * z)")
        -- note: a string, not a lua function
        -- this will be passed to OpenCL kernel
</pre></tr>

</table>

# Installation

* First install torch distro, see [https://github.com/torch/distro](https://github.com/torch/distro).
* Now, do:
```
git clone --recursive https://github.com/hughperkins/cltorch.git
cd cltorch
luarocks make rocks/cltorch-scm-1.rockspec
```
* If you get an error message about some files in EasyCL or clBLAS not existing, then do:
```
git submodule init
git submodule update
```
* If, during `git submodule update` command, it says something about `clBLAS/src` directory already exists, then do:
```
git submodule init
rmdir clMathLibraries/clBLAS/src
git submodule update
```

# Co-existence with cutorch

* You must ensure that cltorch is `require`d after cutorch
* If you get errors about #1 argument to copy should be tensor, but is userdata, then please double-check that cltorch is `required`d after cutorch (they each monkey-patch torch, but since cutorch was written first, it assumes there is no monkey-patch conflict)

# Updating

* Sometimes you might want to do `git pull` to pull in new updates
* If you try this, you might see build errors about EasyCL
* In the future, these might be handled automatically by the build script :-)
* For now, note that two possible issues after a `git pull` are
  * the EasyCL submodule may not have been updated
  * the EasyCL subproject may not have been rebuilt
* To solve these issues, after doing `git pull`, you can do the following, which will ensure the EasyCL submodule is up to date, and will be fully rebuilt, and installed:
```
git submodule update
rm -Rf build/EasyCL
```
* Now you can run the luarocks make command, as above, and hopefully it will work this time :-)
  * if it doesnt, please raise an issue.  It might be easy to fix, but I cant help to fix it, if I dont know about it :-)

# Dependencies

cltorch has the following build dependencies:
* [lua 5.1](http://www.lua.org/versions.html) libraries - used for runtime Kernel templating
* [clBLAS](https://github.com/clMathLibraries/clBLAS) - provides GPU-based matrix operations, such as multiplication
* [EasyCL](https://github.com/hughperkins/EasyCL) - provides an abstraction layer over the low-level OpenCL API
* [clew](https://github.com/martijnberger/clew) - similar to glew, means that cltorch can be loaded without any OpenCL library/runtime being present

At runtime, if you want to call any of the cltorch methods, you will also need:
* OpenCL-compatible GPU
* OpenCL library/driver (normally provided by the GPU vendor)

# Requests for additional operations etc

* Please raise an issue for any operations etc which you particularly need, or you feel are not working for some reason.
* (Ditto for any build errors)

# Unit tests / samples

* There are unit tests at:
  * For ClStorage: [test/cltorch-unit-storage.lua](test/cltorch-unit-storage.lua)
  * For ClTensor: [test/cltorch-unit-tensor.lua](test/cltorch-unit-tensor.lua)
* There's also a sample of interrogating devices at:
  * [test/test-device.lua](test/test-device.lua)

# Guidelines for contributors

You might or might not find [ContributorGuidelines.md](doc/ContributorGuidelines.md) useful.  Not required reading, but it is there if you want to see my own thoughts and ideas on how I am currently approaching cltorch development, and cutorch-porting.

Also, some more technical guidelines on porting, in the [clnn](https://github.com/hughperkins/clnn) repository, at [porting-guidelines.md](https://github.com/hughperkins/clnn/blob/master/doc/porting-guidelines.md).

# Related projects

There is an OpenCL backend for `nn` and `nngraph` at [clnn](https://github.com/hughperkins/clnn).

# Recent changes

* 5th July:
  * fixed some Mac build/load issues, so builds/loads on Mac now (thank you to mlajtos, szagouyko, centime, luo123n, and pdhvip for their enormous help with fixing this :-) )
  * getDeviceProperties and so on now only show GPU and APU devices, ignore pure CPU devices (which pure CPU devices are not supported by cltorch at this time)
* 4th July:
  * `torch.save` and `torch.load` implemented
* 27th June:
  * fixed more bugs involving Tensor copy.  Hopefully should be fixed permanently now :-P
  * added `cltorch.dumpTimings()`, which will dump cumulative timings for various parts of the engine.  It's mostly for usage by maintainers / optimizers.
  * massive optimization for anything involving apply, reduce, reduceall, index etc => this makes the ltsm script at [karpathy/char-rnn](https://github.com/karpathy/char-rnn) run significantly faster when using OpenCL now :-)
* 26th June:
  * add addcmul, and unit test
  * add addcdiv, and unit test
  * added `apply2` and `apply3` as synonyms for `map` and `map2`
  * can use `x`, `y`, `z` instead of `*out`, `*in1` and `*in2`, in `apply`, `map`, etc
  * fix a buffer copy bug (note: implies updating EasyCL, and rebuilding EasyCL, see notes on updating above)
* 25th June:
  * added bernoulli (generates on host-side for now, but I guess this is fast enough for many things?)
* 24th June:
  * added tests for `gather`, and removed some spam
  * added `scatter` (for both tensor or float source)
* 23rd June:
  * Fixed bug where operations such as apply and map on tensors with non-zero offset didnt work correctly (ie, `fill` etc after `narrow` or similar)
  * Added `gather`
* 22nd June:
  * Under the hood:
    * Moved marking a buffer dirty, ie modified on the GPU, from [THClTensorMathBlas.cpp](https://github.com/hughperkins/cltorch/blob/9133fb4f0a23a86c48dcb5dc9cc7d44f44850a3f/lib/THCl/THClTensorMathBlas.cpp#L202) to [THClBlas.cpp](https://github.com/hughperkins/cltorch/blob/9133fb4f0a23a86c48dcb5dc9cc7d44f44850a3f/lib/THCl/THClBlas.cpp#L424)
      * This fixes a bug in [clnn](https://github.com/hughperkins/clnn), where the results of a convolutional layer were not being written back to the output tensor
  * tests pass now on an AMD gpu (actually I managed to scrounge access to a W9100 :-D )
* 21st June:
  * Under the hood:
    * Upgraded new THClKernels class to handle `THClTensorInfo`
    * migrated Reduce, ReduceAll, etc to use THClKernels
    * upgraded EasyCL to handle `uint`, `long`, `ulong`
  * added `cltorch.finish()` and `cltorch.synchronize()`, both do same thing, which is a `clFinish()`, on current device
  * made it possible to require both cutorch and cltorch, as long as one requires cutorch followed by cltorch, in that order
* 20th June:
  * rename new `sub` method to `csub` so doesnt collide with existing `sub`
  * added `cltorch.setTrace(1|0)`, which prints out every allocate or copy of gpu buffers (named 'wrapper's)
  * removed `set` and `get` methods, because cause repeated gpu buffer copy (actually, get not too bad, but does copy whole buffer; set copies whole buffer, repeatedly :-P )
  * modifed `ClStorage.__string__` to first copy whole storage to FloatStorage, once, then convert this to string, rather than using now non-existent `get`
  * `torch.ClTensor{3,5,2}` will now first create this as a `FloatTensor` then call `copy` on this, to convert whole Tensor/Storage to `ClTensor` (avoids repeated `set` calls)
  * added `normall`, ie can do `torch.norm(c)`, `torch.norm(c, exponent)`
  * added `prod`, `prod(1)`, `prod(2)`
  * `max(1)` and `min(1)` now return the indices too, as well as the max.  Ditto for dimension 2.
  * added `:all()` and `:any()`
  * added `:indexFill()`
  * added `:indexCopy()`
  * added `:indexSelect()`
  * added `torch.cumsum(x,2)` and `torch.cumsum(x,1)`
  * added `torch.cumprod(x,2)` and `torch.cumprod(x,1)`
  * Under the hood:
    * created new THClKernels class:
      * handles THClTensor kernel input
      * provides `run` method that takes a dim3 `grid` and `block` input, as for cutorch kernel launches
      * migrated TensorIndexed to use THClKernels
* 19th June:
  * fixed a compile bug in EasyCL, when lua5.2/5.3 header files are present (not tested yet)
  * added `a:sub(b)` method, which does element-wise subtraction of b from a, and puts results in a
  * migrated to new version of EasyCL, with one fewer waitforevents, to try to boost perf a bit
  * added `apply`, `map`, `map2` :-)  (which run on GPU, at full speed)
  * added 2-pass reduceall, ie can do reduceall on much larger tensors now
* 18th June:
  * fixed a bug in clBLAS sger that meant that sger crashed on even tiny 5x5 matrices on nvidia, using either rowmajor or columnmajor :-)  https://github.com/clMathLibraries/clBLAS/pull/109
  * note that you will need to `git submodule update`, and `rm -Rf build/clBLAS`, in order to pick up the new version of clBLAS
  * moved clBLAS initialization code out of inner loops => huge speed boost
  * added `:neg()` operator, which negates the tensor (like `-` but without reallocation, I think)
* 15th-17th June:
  * pow(x,y) no longer returns undefined values for x containing, or being, negative
  * pow(x,y) now uses `pown` when y is an exact integer scalar (ie where (float)((int)y) == y)
  * when no opencl-enabled devices enabled, now raise a THError, with a clear error message, rather than throwing a C++ exception, with no error message output
  * under the hood: added cltorch.getState()
  * renamed libTHCL.so to libTHCl.so
  * added THCl include files to `install` section
  * masked fill works now
  * torch.addr works now
* 15th June:
  * C:t() working
* 14th June:
  * ReduceAll working :-)  For now means: sometensor:sum() works
  * sometensor:sum(1) and sometensor:sum(2) working too now :-)
  * A:min(), A:max() added
  * created unit tests, in [test](test) directory, [cltorch-unit-tensor.lua](test/cltorch-unit-tensor.lua) which pass
* 13th June:
  * added `cltorch.setDevice`/`cltorch.getDevice`, see [test-device.lua](test/test-device.lua) for an example
  * added EasyCL includes to EasyCL install section, to remove build errors with "EasyCL.h" not found, etc


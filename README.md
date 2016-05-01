# cltorch

An OpenCL backend for [torch](http://torch.ch/).

## What is this?

It's a high-performance matrix library for OpenCL, that runs on your GPU(s) harnessing the massive computational capacity that they provide.

Most of the standard operations in [torch](https://github.com/torch/torch7) are supported.  If there are any missing that you need, please raise an issue.

## What's working

Most things really :-)  Detailed description at [ImplementedDetails.md](doc/ImplementedDetails.md).  Please don't hesitate to raise an issue for anything that's missing that you would like to see added.

## Installation

IMPORTANT!  THIS HAS CHANGED.  Please install a specific Torch distro, as described below.  Simply doing `luarocks install cltorch` is no longer supported

### Pre-requisites

* python 2.7 installed: `python` command should point to python 2.7, during build

### Procedure

Please proceed as follows:
```
git clone --recursive https://github.com/hughperkins/distro -b distro-cl ~/torch-cl
cd ~/torch-cl
bash install-deps
./install.sh
```
Thats it!  To test, you can do for example:
```
source ~/torch-cl/install/bin/torch-activate
luajit -l cltorch -e 'cltorch.test()'
```
Actually, it will also install clnn, eg the following should be working ok now too:
```
luajit -l clnn -e 'clnn.test()'
```

## Gotchas

* On MAC OS X, LD_LIBRARY_PATH is typically not used to locate dependent libraries.  It means, cltorch might not load for you.  A temporary hack, until rpath is patched into the build, is to do:
```
ln -s ~/torch-cl/install/lib ~/lib
```

## Updating

Please do *NOT* use any of: `luarocks install nn`, `luarocks install torch`, `luarocks install cltorch`, `luarocks install clnn`,
`luarocks install cutorch`, or `luarocks install cunn`.  This will break your installation, and is not supported.  The supported update method is:
```
cd ~/torch-cl
git pull
git submodule update --init --recursive
./install.sh
```
If any errors like `fatal: reference is not a tree`, you have two options:
* easy option: remove ~/torch-cl completely, reinstall
* harder, hacky option:
  * in the error message, you should see which submodule is broken.  Let's say it is `extra/nn`
  * so edit `.git/modules/extra/nn/config`, and in the `url` part, change `torch` to `hughperkins`
  * if it's not `extra/nn`, then modify the path of this file appropriatel
  * that's it!
  * now rerun `git submodule update --init --recursive`, and the updates should pull down ok (otherwise raise an issue)

## Requests for additional operations etc

* Please raise an issue for any operations etc which you particularly need, or you feel are not working for some reason.
* (Ditto for any build errors)

## Unit tests / samples

Simply run:
```
luajit -l cltorch -e 'cltorch.test()'
```

These tests should systematically run clean.  They do on the systems I've tested against.  If they don't, it's a bug.  Please raise an issue, including your operating system, graphics card, 32-bit/64-bit, all full logs, and anything else you can think of.  Also output of `th -l cltorch -e 'cltorch.about()'` please.

## cltorch-specific features

The following features are either cltorch-specific, or do not exist in cutorch:

|feature|in torch?|in cutorch?|
|---|---|---|
| apply/map/map2 | Yes |  |
| optimization tools | | |
| point tensors | | |
| custom user kernels | Not applicable | |

### apply/map/map2

`apply`, `map`, `map2` exist in torch, but how to make them work on the GPU?  Cannot just pass in lua functions, at least not a the moment.

What we do is, you can provide opencl code directly to apply, map and map2, as a string expression.  This will run on the gpu, at full speed.  Examples, for `x`, `y`, `z` being identically sized `torch.ClTensor`s:
```
x:apply("x = sqrt(x + 3.5)")
x:map(y, "x = 1000 * x + y * 10")
x:map2(y, z, "x = sqrt(1000 * x + y * 10 + z * z)")
```
* note that the variables in the OpenCL string expression must be named as above, ie `x`, `y`, `z`.  For convenience, these were named the same as the tensors in the example.  If the tensors have different names, please continue to use `x`, `y`, `z` in the expressions, eg:
```
a:apply("x = sqrt(x + 3.5)")
a:map(b, "x = 1000 * x + y * 10")
a:map2(b, c, "x = sqrt(1000 * x + y * 10 + z * z)")
```

### Optimization tools

Following tools are available to aid with optimization:

|Method|Description|
|------|---------|
|`cltorch.setProfiling(1)` |  turn on opencl kernel profiling |
|`cltorch.dumpProfiling()` | dump opencl kernel profiling timings since last call|
|`cltorch.setEnableTiming(1)`  | enable collection of cumulative wall-clock timings for cltorch code |
|`cltorch.dumpTimings()`  | dump cumulative wall-clock timings for cltorch code |
|`cltorch.setTrace(1)` | print all gpu buffer allocations and copies between host/gpu |

#### OpenCL Profiling

OpenCL natively provides facilities to measure the execution time of kernels, without needing to call `cltorch.finish()` or similar first, using [clGetEventProfilingInfo](https://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clGetEventProfilingInfo.html).  In cltorch, you dont need to know how this works ;-)  Simply call, at the start of your code:

```
cltorch.setProfiling(1)
```
Then, after running the piece of code under scrutiny, simply call:
```
cltorch.dumpProfiling()
```
Timings are cumulative across multiple calls to the same kernel.

#### DumpTimings

This uses the wall-clock times to measure the elapsed time in different sections of cltorch code.  The way it works is, each time the cltorch c++ code calls `StatefulTimer::instance()->timeCheck("some status")`, the wall-clock time since the last call to `->timeCheck()` will be added to the cumulative time for `some status`.  You can pass any status as a string.  Then, after running the piece of code under the scrutiny, in your Lua program, simply call `cltorch.dumpTimings()` to dump these cumulative timings.

Update: please first call `cltorch.setEnableTiming(true)` to enable collection of timing information.  This is global across all devices.

#### GPU buffer allocations and copies

You can log all GPU buffer allocations, copies to host, and copies to GPU device.  Simply call:
```
cltorch.setTrace(1)
```
Any buffer allocations, and copies between host and device, will now be printed to stdout.

### Point tensors: reduce pipeline stalls

Point tensors help to eliminate pipeline stalls associated with ReduceAll operations such as `sometensor:sum()`.  Why does `:sum()` cause pipeline stalls, and how do point tensors eliminate this source of stalls?

If we send a single instruction (a kernel) to the gpu, there will be some latency whilst the instruction arrives at the gpu, and starts running, and some more latency after the calculations have finished, whilst the results are retrieved back from the GPU.  Maybe we send:

```
a:add(1)
```
We can draw a picture of what happens.  Time is towards the right.  GPU is at the top.  CPU at the bottom:

![gpu single instruction](doc/img/singlegpuoperation.png)

But we can send lots of instructions, without waiting for the earlier ones to finish. Maybe we do:
```
a:add(b)
a:mul(3)
b:mul(a)
c:add(a)
```
This might look like this, we dont have to wait for the previous instruction to finish:
![gpu pipeline](doc/img/gpupipelinemultiple.png)

But now imagine what happens if we process the following instruction:
```
a:div(a:sum())
```
- a:sum() is going to take the sum of all the elements in a
- a:div(a:sum()) is then going to divide all the elements of a by this sum
- it looks innocent enough
- but notice that we cannot send the `a:div` instruction until the `a:sum()` results have come back
- so we have to wait for `a:sum()` to finish processing, and for the results to come back, before we can continue

Looks like this:
![gpu stall](doc/img/reduceall_pipelinestall.png)

*Classic reduceall => Massive pipeline stall*

*Point tensors eliminate this*.  When we do the reduceall, the `:sum()` operation, we keep the results on the gpu, like this:
```
c = torch.Tensor(20,30):uniform():cl() -- create a tensor on the GPU
res = torch.ClTensor()                 -- create a point tensor on the GPU
res:sum(c)                             -- sum c, and keep the result in res, on the GPU
```
`res` is a point tensor.  It has zero dimensions.  It contains a single scalar float.  It stays on the GPU.  We can feed it into other operations as follows:
```
c:div(res)  -- divide c by res
```
We can send this instruction straight away, even before the first `:sum(c)` instruction has arrived at the GPU.  So, no more stall.

By the way, it's possible to print the value of a point tensor, by printing it, or calling the `:s()` operator.  Normally you wouldnt do this except during debugging though, since obviously this will need to wait for the gpu operation to finish, and for the data to come all the way back from the GPU :-)

### Custom user kernels

Custom user kernels let you run OpenCL code directly from Torch Lua!  Of course, you can already do this with `apply`, `map`, and `map2`, see above.  But now you can provide whole kernel functions, and other functions, and pass ClTensors into these kernels!

Example of how to use:

```
require 'cltorch'

k = torch.ClKernel({input={nElements='int', input='torch.ClTensor'},output={output='torch.ClTensor'},src=[[
   int linearId = get_global_id(0);
   if(linearId < nElements) {
     output_data[linearId] = input_data[linearId] + 3.0f;
   }
]]})
print('k', k)
k:print()

x = torch.ClTensor({3,5,2})
y = torch.ClTensor({6,4,2})
print('x before\n', x)
print('y before\n', y)

k:run({nElements=3, input=x, output=y})

print('y after\n', y)
```

Output from this example:
```
Using Intel platform: Intel Gen OCL Driver
Using device: Intel(R) HD Graphics BroadWell U-Processor GT2
k	torch.ClKernel
Original source
===============
   int linearId = get_global_id(0);
   if(linearId < nElements) {
     output_data[linearId] = input_data[linearId] + 3.0f;
   }


Generated source
================
typedef struct THClTensorInfoCl {
  unsigned int sizes[25];
  unsigned int strides[25];
  int offset;
  int dims;
} TensorInfoCl;



kernel void user_kernel(
    global struct THClTensorInfoCl *input_info, global float * input_data,
    int nElements,
    global struct THClTensorInfoCl *output_info, global float * output_data
) {
   int linearId = get_global_id(0);
   if(linearId < nElements) {
     output_data[linearId] = input_data[linearId] + 3.0f;
   }

}

x before
	 3
 5
 2
[torch.ClTensor of size 3]

y before
	 6
 4
 2
[torch.ClTensor of size 3]

y after
	 6
 8
 5
[torch.ClTensor of size 3]

```

If you want, you can specify the number of workgroups, and the workgroupsize, yourself:
```
k:run({nElements=3, input=x, output=y}, {numWorkgroups=10, workgroupSize=32}
```

## Co-existence with cutorch

* It is possible to load cutorch and cltorch at the same time, if you wish
* If you do this, please load cutorch first, and then load cltorch second
* If you get errors about #1 argument to copy should be tensor, but is userdata, then please double-check that cutorch is `required`d before cltorch (they each monkey-patch torch, but since cutorch was written first, it assumes there is no monkey-patch conflict)

## Third-party libraries

cltorch uses the following libraries. These are automatically built as part of cltorch build process:
* [clBLAS](https://github.com/clMathLibraries/clBLAS) - provides GPU-based matrix operations, such as multiplication
* [EasyCL](https://github.com/hughperkins/EasyCL) - provides an abstraction layer over the low-level OpenCL API
* [clew](https://github.com/martijnberger/clew) - similar to glew, means that cltorch can be loaded without any OpenCL library/runtime being present

At runtime, if you want to call any of the cltorch methods, you will also need:
* OpenCL-compatible GPU
* OpenCL library/driver (normally provided by the GPU vendor)

## Guidelines for contributors

You might or might not find [ContributorGuidelines.md](doc/ContributorGuidelines.md) useful.  Not required reading, but it is there if you want to see my own thoughts and ideas on how I am currently approaching cltorch development, and cutorch-porting.

Also, some more technical guidelines on porting, in the [clnn](https://github.com/hughperkins/clnn) repository, at [porting-guidelines.md](https://github.com/hughperkins/clnn/blob/master/doc/porting-guidelines.md).

## Related projects

There is an OpenCL backend for `nn` and `nngraph` at [clnn](https://github.com/hughperkins/clnn).

## Recent changes

* 31 April 2016:
  * Re-applied:
    * 27 March 2016:
      * migrated from clBLAS 2.4 to clBLAS 2.11/develop.  This migration is not set in stone, depends on how well that works.  However, there is a
    [bug in 2.4 for certain configurations of matrix multiplication](https://github.com/clMathLibraries/clBLAS/issues/246), and its not obvious how to fix that, so maybe using 2.11/develop is the easiest way forward?
* 30 April 2016:
  * rolled back to as of 3 March 2016, to use specific torch release, so it doesnt keep changing whilst I'm at work :-)
* 3 March 2016:
  * runs on Mac OS X, without needing `LD_LIBRARY_PATH`, ie [RPATH](https://cmake.org/Wiki/CMake_RPATH_handling) works now.  Hopefully :-)
* 3rd January, 2016:
  * created Mac build on Travis, https://travis-ci.org/hughperkins/cltorch , which passes (at time of writing)
* 27th December:
  * added FFI functions `:data()` and `:cdata()`, which means that Element Research's [rnn](https://github.com/element-research/rnn) now works with `clnn`
* 23rd October:
  * removed `:csub()` and `:neg()` from the "cltorch-specific features" section, since integrated into torch now :-) [pull request 392](https://github.com/torch/torch7/pull/392)
* 3rd October:
  * Added `:mean()` and `:mean(d)`
  * Added `:atan2(x,y)`
  * Added `x:sign()` and `torch.sign(x)`
  * Added `norm(...)`
* 20th September:
  * Ported fix to `addcdiv` and `addcmul` reshape from cutorch commit [59a1cb05](https://github.com/torch/cutorch/commit/59a1cb05745d7b03d896d7a950c4845c9eebb73f)
  * Added ClStorage:__index() and ClTensor:__index()
  * Added ClStorage:__newindex() and ClTensor:__newindex()
* 19th September:
  * Added guards around many functions, so that c++ exceptions are converted to torch errors now, and display something more meaningful than just 'c++ exception' :-P
    * Please feel free to raise an issue for any exceptions which are not guarded yet

[OlderChanges.md](doc/OlderChanges.md)


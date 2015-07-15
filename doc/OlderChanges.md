# Older changes

This page contains older changes, that have been moved from the [Recent Changes](https://github.com/hughperkins/cltorch#recent-changes) section on the main page.

For the most recent changes please see [Recent Changes](https://github.com/hughperkins/cltorch#recent-changes)

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


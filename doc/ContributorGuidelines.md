# Contributor Guidelines

This doc describes some of the things I use as guidelines when writing cltorch

Cutorch is an awesome, excellent high-performance implementation.  Whenever a functionality is available in cutorch, I think that porting the cutorch implementation to cltorch is going to tend to have several advantages:
* can integrate future improvements from cutorch
* development will tend to be much faster.  Probably at least 4-10 times to port something existing, than to write it from scratch
* correctness will likely be high, not too many logical bugs
* performance will likely be reasonable

Some things are not directly portable.  Notable examples are:
* thrust
* cuda templated kernels

Thrust is a cuda-specific library.  There is something similar for OpenCL, which is VexCL. For now, I prefer not to use VexCL, because it uses Boosts, which, rightly or wrongly, I somehow feel is a bit of a sledgehammer-to-crack-a-nut, really hard to build on Windows.  My last experience with boost was probably 11 years ago, so it might have changed :-P

So, generally, for thrust, there are a few options I've used up till now:
* for the reduceall method, it turned out there was a thrust-free implementation in `goodies` branch of cutorch, so I ported that across, and seems to work great :-)
* for the MSECriterion implementation in `clnn`, I simply wrote the operations on the lua side, using the cltorch `pow` and so on implementations.  I'm pretty sure performance will be ok
* thrust is used all over the place.  Some creativity will be required.  Please get in touch if you come across a new situation, so we can discuss together.  I mean, you dont have to, but maybe could be a good idea :-)

Cuda C++ templates dont exist in OpenCL, at least not in the OpenCL 1.1 implementation I'm targeting.  Interestingly, OpenCL kernels are compiled at runtime, which some might see as a disadvantage, but it actually gives us a lot of flexibility.  And you can see that cltorch loads really quickly, whereas cutorch spends a while caching every possible kernel, when loaded.  This has good or bad points, for either really.

So... for the kernel templates, I quite like the compile-at-runtime approach.  I'm using a lua engine to instantiate the kernel templates for cltorch at runtime, at the point where they are first needed.  They are reused across multiple calls.  Nvidia caches these compiled kernels to disk, where they can be reused almost immediately, even after restarting the process.

# Porting utilities

There is a python script in `src/util` called 'port.py'.  It can help do a first-cut port of files, or facilitate meld of existing files.  Run it as follows:
* in the parent directory of 'cltorch', clone the 'cutorch' repository
  * currently, it expects the cutorch 'goodies2' branch to be cloned into `cutorch-goodies2` directory, but obviously you can hack port.py a bit to change the exact directory
* from cltorch directory, run `python src/util/port.py`
* A first-cut port of the files in ../cutorch-goodies2 will pop out in the `port` directory
* .cuh files will become .h files
* .cu files will become .cpp files
* any kernels and kernel functions will plausibly be split into .cl files (with the same basename as the original .cu or .cuh file)

# Adding original functionality, not present in cutorch

For now, I haven't really come across this situation :-P  The only brush I had with this was considering adding ClByteTensors, but for now, I've shelved the idea of implementing those initially in cltorch.

I think that on the whole, for now, cutorch is the 'reference' implementation, and will probably remain so for a while.  There is a whole team of incredibly talented, hard-working, motivated individuals maintaining, and improving cutorch. For the foreseeable future, I think cltorch will be following cutorch, though you never know :-)

Therefore, on the whole, if I dont have to implement the original functionality myself, my recommendation would be: first add it to the cutorch side, then port it across to cltorch.

On the other hand, in fairness, if it was me, I'd probably do it on the cltorch side, and plausibly in a way totally unlikely to encourage back-port into cutorch :-P  So, anyway, if you want to implement something original in cltorch, perhaps you can discuss with me, and on the torch7 newsgroup?

# Operator classes

On the subject or original functionality, or at least, original implementations, in cutorch, operators, ie AddOp etc, are structs, which are injected directly into the nvcc compiler.  In OpenCL, we dont have c++ in the kernels, it should be C99.  So... well, that doesnt mean we couldnt use structs actually but ... we cant just take a struct from our .cpp/.h file and inject it into a kernel.  We need to provide it as a text file.  Again, thinking this through as I write, there's no particular reason why we cant provide structs to the OpenCL kernels.

Anyway.... rightly or wrongly :-P  what I've done for now is to change the operator structs into C++ classes, which derive from HasOperator1, HasOperator2, HasOperator3 and/or HasScalars.  These are interfaces. HasOperator2 has a function called 'operator2()', which returns a string.  The string will be injected into our OpenCL kernel templates.

I think it works quite nicely, and it's easy to convert the structs into classes, and visa versa, though it is admittedly a slight deviation from the cutorch design.

# cogapp

Oh yes, by the way, I'm using [cogapp](https://bitbucket.org/ned/cog) to help do some of the templating.  It needs a python environment.  By default, it doesnt run, but if you want to modify any of the cl files, you'll need to rerun stringify.  To get this to work:
* make sure you have python available
* cd into `build` directory, and do `ccmake ..`
* set option `DEV_RUN_COG` to `ON`
* and do `configure` then `generate`
* => from now on, cogapp will run automatically, when you build, and reimport the .cl files into the corresponding .cpp file

# Possible future libraries that might be useful to use

* viennacl: might provide competitive GEMM
* clBLAS: might provide competitive GEMM
* blas.compute: might provide useful implementations of sort, scan, (maybe also reduce, but we already have custom torch implementation
 for reduce)


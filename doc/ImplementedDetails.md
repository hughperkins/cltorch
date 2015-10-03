# Implemented Details

Here's a list of the functionalities in [torch](https://github.com/torch/torch7), and the status of their implementation in [cltorch](https://github.com/hughperkins/cltorch).

## Import

<table>

<tr><td>Component<td>Status<td>Examples of what works now</tr>

<tr><td><pre>require 'cltorch'</pre> <td> works <td><pre>require 'cltorch'</pre></tr>

</table>

## Device information and control

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

## From [storage.md](https://github.com/torch/torch7/blob/master/doc/storage.md)

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

## From [tensor.md](https://github.com/torch/torch7/blob/master/doc/tensor.md)

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

<tr><td>Querying elements<td>Done<td><pre>
c = torch.ClTensor(3,2,5)
c[2,1,4] = 23
assert(c[2,1,4] == 23)
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

## From [random.md](https://github.com/torch/torch7/blob/master/doc/random.md)

<table>

<tr><td>Component<td>Status<td>Examples of what works now</tr>

<tr><td>uniform, etc <td>2%<td><pre>For now, you would need to do for example:
torch.Tensor(5,3):uniform():cl()
torch.Tensor(5,3):bernoulli() -- works now, but basically generates on host side
     -- then copies to gpu
</pre></tr>
</table>

## From [serialization.md](https://github.com/torch/torch7/blob/master/doc/serialization.md)

<table>

<tr><td>Component<td>Status<td>Examples of what works now</tr>

<tr><td>Serialization <td>50%<td><pre>
torch.save('filename.dat', torch.ClTensor{3,5,2})
a = torch.load('filename.dat')
</pre></tr>
</table>

## From [maths.md](https://github.com/torch/torch7/blob/master/doc/maths.md)

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

a:cmin(b)
a:cmax(b)
a:cmin(0.6)
a:cmax(0.6)
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
B = torch.ClTensor{{3,5,2},{4,5,6}}
A:sum()
A:sum(2)
A:sum(1)
A:mean()
A:mean(1)
A:mean(2)
A:sign()
A:norm()
A:norm(1)
A:norm(1,2)
A:norm(1,1)
torch.sign(A)
torch.atan2(A, B)
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

-- arbitrary apply/map
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

-- Point tensors
c = torch.ClTensor({3,4,7})
a = torch.ClTensor()
a:sum(c) -- a is still a ClTensor, stays on GPU
a:prod(c)  -- a is still a ClTensor, on GPU
a:min(c)   -- a is ClTensor, on GPU
a:max(c)   -- a is ClTensor, on GPU
a:all(c)   -- a is ClTensor, on GPU
a:any(c)   -- a is ClTensor, on GPU
c:add(a) -- can pass a into :add
c:csub(a) -- ... or csub
c:mul(a)  -- ... or mul
c:div(a)  -- ... or div
c:fill(a) -- ... or fill
c:lt(a)  -- or any of the logical operators:
c:gt(a)
c:eq(a)
c:ne(a)
c:le(a)
c:ge(a)

-- optimization tools:
cltorch.setProfiling(1)  -- turn on opencl kernel profiling
cltorch.dumpProfiling()  -- dump opencl kernel profiling 
                         -- timings since last call
cltorch.dumpTimings()    -- dump cumulative wall-clock timings
                         -- for cltorch code
cltorch.setTrace(1)      -- print all gpu buffer allocations
                         -- and copies between host/gpu

-- misc
cltorch.about() -- dump version/build information
</pre></tr>

</table>



-- this is mostly for trying things
-- for unit tests, see cltorch-unit-tensor.lua

print("running require cltorch...")
require 'cltorch'
print("... require cltorch done")

print(cltorch.getDeviceProperties(cltorch.getDevice()).deviceName)

function torch.traceon(state)
  -- nop
end

if os.getenv('TRACEON') ~= nil then
  -- following command needs https://github.com/hughperkins/torch7.git, branch
  --                         add-trace
  -- lets one see storage allocs/copies
  function torch.traceon(state)
    torch.setstoragetrace(state)
  end
end

function myprint(a,b)
  torch.traceon(0)
  print(a,b)
  torch.traceon(1)
end

torch.traceon(1)

if true then
a = torch.Tensor{3,5,2}
print('a\n', a)

c = torch.ClTensor{7,4,5}
print('c1\n', c)

c = torch.ClTensor(3,2)
print('c2\n', c)

a = torch.Tensor{2,6,9}
c = a:cl()
print('c3\n', c)

b = c:float()
print('b\n', b)

c = torch.ClTensor{{3,1,6},{2.1,5.2,3.9}}
print('c4', c)

d = torch.ClTensor(2,3)
print('d', d)

d:copy(c)
print('d2', d)

b = torch.Tensor{{4,2,-2},{3.1,1.2,4.9}}
b[1][2] = 2.123
print('b2\n', b)

c = torch.ClTensor{{4,2,-2},{3.1,1.2,4.9}}
c[1][2] = 2.123
print('c5\n', c)

b[1][2] = 5.432
c:copy(b)
print('c6\n', c)

-- =============================

d = torch.ClTensor{{3,5,-2},{2.1,2.2,3.9}}
c = torch.ClTensor{{4,2,-1},{3.1,1.2,4.9}}
c:add(d)
print('c2\n', c)

a = c:float()
b = d:float()
a:cmul(b)
print('a', a)

c:cmul(d)
print('c2\n', c)

c:cdiv(d)
print('c2\n', c)

c = c + d
print('c3\n', c)

c = c - d
print('c3\n', c)

c:abs()
print('c3\n', c)
c:sqrt()
print('c3\n', c)
for _,name in ipairs({'log','exp', 'cos', 'acos', 'sin', 'asin',
   'atan', 'tanh', 'ceil', 'floor', 'abs', 'round'}) do
  print('name', name)
  c = torch.ClTensor{{4,2,-1},{3.1,1.2,4.9}}
  loadstring('c:' .. name .. '()')()
  print('c3\n', c)
end

c[2][1] = d[2][1]
c[1][2] = d[1][2]
for _,name in ipairs({'lt','le','gt','ge','ne','eq'}) do
  print('name', name)
  print(loadstring('return torch.' .. name .. '(c,d)')())
end

print('c\n', c)
print('torch.add', torch.add(c,3))
c:add(3)
print('c\n', c)
c:mul(3)
print('c\n', c)
c:div(2)
print('c\n', c)
c = torch.mul(c, 4)
print('c\n', c)
c = torch.div(c, 3)
print('c\n', c)
c = c / 2
print('c\n', c)
c = c * 1.5
print('c\n', c)
c = c + 4
print('c\n', c)
c = c - 5
print('c\n', c)

for _,name in ipairs({'lt','le','gt','ge','ne','eq'}) do
  print('name', name)
  print(loadstring('return c:' .. name .. '(5)')())
end

print('c\n', c)
print(torch.pow(2,c))
c:pow(2)
print('c\n', c)
print(torch.pow(c,2))

print('c\n', c)
print(torch.clamp(c, 50, 100))
c:clamp(50, 100)
print('c\n', c)

print(torch.cpow(c,d))
print(torch.cdiv(c,d))
print(-c)

-- print(c:t())
end
if true then
A = torch.ClTensor{{1,2,-1},
                   {3,4,0}}
B = torch.ClTensor{{0,1},
                   {1,2},
                   {4,5}}
print('A\n', A)
print('B\n', B)
print(torch.mm(A,B))
end
if true then
print(torch.mm(A:float(), B:float()))

C = torch.ClTensor{{0,0},{0,0}}
C:mm(A,B)
print(C)

print( A * B )
C:fill(1.345)
print('C\n', C)


s = torch.LongStorage{3,2}
print('s\n', s)
--C = cltorch.ones(s)
--print('C\n', C)
C:zero()
print('C\n', C)


--C:reshape({4,1})
--print('C\n', C)

end

if true then
v1 = torch.ClTensor{3,5,1}
v2 = torch.ClTensor{2,4,8}
--print(v1 * v2)
print(torch.dot(v1,v2))

fv1 = torch.FloatTensor{3,5,1}
fv2 = torch.FloatTensor{2,4,8}
print(fv1*fv2)


print(torch.dot(v1,v2))


print(torch.ClTensor.zeros(torch.ClTensor.new(), 3, 5))
print(torch.ClTensor.ones(torch.ClTensor.new(), 3, 5))
-- print(torch.ClTensor.eye(torch.ClTensor.new(), 3))
-- print(torch.ClTensor.diag(torch.ClTensor{{3,5,4},{2,3,4},{7,6,5}}))


print(torch.mv(A,v1))
end


-------------------

if true then

A = torch.ClTensor{{3,5,2},{4,5,6}}
print('A\n', A)
print('A:sum()', A:sum())

print(torch.Tensor.__eq)
function torch.Tensor.__eq(self, b)
  print('self', self)
  diff = self:ne(b)
  print('diff', diff)
  sum = diff:sum()
  print('sum', sum)
  if sum == 0 then
    return true
  else
    return false
  end
end

function torch.ClTensor.__eq(self, b)
  print('self', self)
  diff = self:ne(b)
  print('diff', diff)
  sum = diff:sum()
  print('sum', sum)
  if sum == 0 then
    return true
  else
    return false
  end
end

--print(torch.Tensor({3,5,2}) == torch.Tensor({3,5,2}))
--print(torch.Tensor({{3,5,2},{4,5,6}}) == torch.Tensor({{3,5,2},{4,5,6}}))

print(torch.ClTensor({{3,5,2},{4,5,6}}) == torch.ClTensor({{3,5,2},{4,5,6}}))
--print('end')
end

if false then
A = torch.ClTensor{{3,2,4},{9,7,5}}
print('A\n', A)
print('A:sum(2)', A:sum(2))
print('A:sum(1)', A:sum(1))
print('A:max()', A:max())
print('A:min()', A:min())
print('torch.max(A,1)', torch.max(A,1))
print('torch.max(A,2)', torch.max(A,2))
end

if false then
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
end

if false then
  c = torch.ClTensor{{3,1,6},{2.1,5.2,3.9}}
  c:fill(1.345)
  print('c\n', c)
  c:zero()
  print('c\n', c)
  c:resize(3,2)
  print('c\n', c)
  c:resize(2,2)
  print('c\n', c)
  c:resize(2,4)
  print('c\n', c)
  l = torch.LongStorage{3,3}
  c:resize(l)
  print('c\n', c)
  d = torch.ClTensor(2,2)
  print('d\n', d)
  d:resizeAs(c)
  print('d\n', d)
end

if false then
  C = torch.ClTensor{{3,2,4},{9,7,5}}
  A = C:float()
  print('C\n', C)
  D = C:reshape(3,2)
  print('D\n', D)
  B = A:reshape(3,2)
  print('B\n', B)
  print('C\n', C)
  print(C:t())
end

if false then
  C = torch.ClTensor{{3,2},{9,7}}
  D = torch.ClTensor{{3,1,7},{3,2,4}}
  E = torch.ClTensor{{3,1},{2,9},{3,2}}
  print(torch.addmm(C,D,E))

  c = torch.ClTensor{3,2}
  D = torch.ClTensor{{3,1,7},{3,2,4}}
  e = torch.ClTensor{3,1,2}
  print(torch.addmv(c,D,e))

  C = torch.ClTensor{{3,1,7},{3,2,4},{8,5,3}}
  d = torch.ClTensor{3,2,5}
  e = torch.ClTensor{3,1,2}
  print(torch.addr(C,d,e))
  print(torch.addr(C:float(), d:float(), e:float()))
end

if false then
  E = torch.ClTensor{{3,1},{2,9},{3,2},{7,8},{6,4}}
  print('E\n', E)
  F = E:narrow(1,2,3)
  print('F\n', F)
  F:fill(7)
  print('F\n', F)
  print('E\n', E)

  E = torch.ClTensor{{3,1},{2,9},{3,2},{7,8},{6,4}}
  print('E\n', E)
  F = E:sub(2,3,2,2)
  print('F\n', F)
  F:fill(0)
  print('F\n', F)
  print('E\n', E)

  E = torch.ClTensor{{3,1},{2,9},{3,2},{7,8},{6,4}}
  print('E\n', E)
  F = E:select(1,2):fill(99)
  print('F\n', F)
  print('E\n', E)

  x = torch.ClTensor(5, 6):zero()
  print('x\n', x)
  x[{ 1,3 }] = 1
  print('x\n', x)
  x[{ 2,{2,4} }] = 2 
  print('x\n', x)
  x[{ {},4 }] = -1
  print('x\n', x)
  x[{ {},2 }] = torch.range(1,5) 
  print('x\n', x)
  x[torch.lt(x,0)] = -2
  print('x\n', x)

end

if false then
--    bias:fill(0.1)
--    addBuffer:fill(0.1)
  print('bias', bias)
  print('bias.storage', bias:storage())
  output:addr(1,addBuffer,bias)
--      self.output:addr(1, self.addBuffer, self.bias)
  print('output\n', output)

  A = torch.Tensor(5,3):uniform()
  B = torch.Tensor(5,3):uniform()
  print('Res', torch.cmul(A,B))
  print('ResCl', torch.cmul(A:clone():cl(), B:clone():cl()))

  print('pow', torch.pow(A,2))
  print('pow cl', torch.pow(A:clone():cl(),2))

  print('- op', - A)
  print('- op', - A:clone():cl())

  Aclneg = A:clone():cl()
  Aclneg:neg()
  print('Aclneg', Aclneg)

  print('torch.add(A,B)', torch.add(A,B))

  Acladd = A:clone():cl()
  Acladd:add(B:clone():cl())
  print('Acladd', Acladd)

   print('A-B', A - B)

  Aclsub = A:clone():cl()
  Aclsub:sub(B:clone():cl())
  print('Aclsub', Aclsub)

  addBuffer = torch.Tensor(128):fill(0.1):cl()
  bias = torch.Tensor(10):fill(0.1):cl()
  output = torch.Tensor(128,10):fill(0.1):cl()

  C = torch.ClTensor(128,10)
  D = torch.ClTensor(128,10)
  C:fill(3)
  D:fill(1)
  print(C - D) 

end

if true then
   A = torch.Tensor(3,2):uniform()
   print('A\n', A)
   A:apply(function(x) return x + 3 end)
   print('A\n', A)

   C = A:clone():cl()

   print('C\n', C)
   C:apply("*out = sqrt(*out + 3.5)")
   print('C\n', C)

   A = torch.Tensor(3,2):uniform()
   B = torch.Tensor(3,2):uniform()
   print('A\n', A)
   print('B\n', B)
   Acopy = A:clone()
   Acopy:map(B, function(a, b) return 1000 * a + b * 10 end)
   print('A\n', Acopy)

   Acl = A:clone():cl()
   Bcl = B:clone():cl()

   Acopycl = Acl:clone()
   Acopycl:map(Bcl, "*out = 1000 * *out + *in1 * 10")
   print('Acopycl\n', Acopycl)

   A = torch.Tensor(3,2):uniform()
   B = torch.Tensor(3,2):uniform()
   C = torch.Tensor(3,2):uniform()
   print('A\n', A)
   print('B\n', B)
   print('C\n', C)
   Acopy = A:clone()
   Acopy:map2(B, C, function(a, b, c) return 1000 * a + 100 * b + c * 10 end)
   print('A\n', Acopy)

   Acl = A:clone():cl()
   Bcl = B:clone():cl()
   Ccl = C:clone():cl()

   Acopycl = Acl:clone()
   Acopycl:map2(Bcl, Ccl, "*out = 1000 * *out + 100 * *in1 + *in2 * 10")
   print('Acopycl\n', Acopycl)

  A = torch.Tensor(28*28*1280,10):uniform()
--  A:fill(2.5)
  print(A[100][5])
  A = A + 2
  print(A[100][5])
  print(torch.sum(A))

  Acl = A:clone():cl()
  print('torch.sum(Acl)\n', torch.sum(Acl))
end

if os.getenv('PROTOTYPING') ~= nil then
  A = torch.Tensor(28*28*1280,10):uniform()
  print('numel A', torch.numel(A))
  Acl = A:cl()
--  torch.setstoragetrace(0)
  print('torch.numel(A:cl())', torch.numel(Acl))

  E = torch.ClTensor{{3,1},{2,9},{3,2},{7,8},{6,4}}
  myprint('E\n', E)
  F = E:narrow(1,2,3)
  myprint('F\n', F)
  F:fill(7)
  myprint('F\n', F)
  myprint('E\n', E)

  E = torch.ClTensor{{3,1},{2,9},{3,2},{7,8},{6,4}}
  myprint('E\n', E)
  F = E:sub(2,3,2,2)
  myprint('F\n', F)
  F:fill(0)
  myprint('F\n', F)
  myprint('E\n', E)

--  E = torch.ClTensor{{3,1},{2,9},{3,2},{7,8},{6,4}}
--  myprint('E\n', E)
--  F = E:select(1,2):fill(99)
--  myprint('F\n', F)
--  myprint('E\n', E)

--  x = torch.ClTensor(5, 6):zero()
--  myprint('x\n', x)
--  x[{ 1,3 }] = 1
--  myprint('x\n', x)
--  x[{ 2,{2,4} }] = 2 
--  myprint('x\n', x)
--  x[{ {},4 }] = -1
--  myprint('x\n', x)
--  x[{ {},2 }] = torch.range(1,5) 
--  myprint('x\n', x)
--  x[torch.lt(x,0)] = -2
--  myprint('x\n', x)

--   print('C\n', C)
--   C:apply("*out = sqrt(*out + 3.5)")
--   print('C\n', C)


--  input = torch.ClTensor{3,5,2}
--  output = torch.ClTensor()
--  weight = torch.ClTensor{{0.2, -0.2, 0.3},
--                        {0.4,-0.1, -0.5}}
--  bias = torch.ClTensor{0.1, -0.2}
--  output:resize(bias:size(1))
--  output:copy(bias)
--  output:addmv(1, weight, input)

--  this fails currently, needs 2stage reduceall:
--  C = torch.ClTensor(128,10)
--  D = torch.ClTensor(128,10)
--  C:fill(3)
--  D:fill(1)
--  print(C - D) 

--  E = torch.ClTensor{{3,1},{2,9},{3,2},{7,8},{6,4}}
--  F = torch.expand(E, 2)
--  print('F\n', F)
--  E[1][2] = 51
--  print('F\n', F)

--  x = torch.range(1,12):double():resize(3,4):cl()
--  print('x\n', x)
--  mask = torch.ByteTensor(2,6):bernoulli():cl()
--  print('mask\n', mask)
--  y = x:maskedSelect(mask)
--  print('y\n', y)
--  z = torch.ClTensor()
--  z:maskedSelect(x, mask)
--  print('z\n', z)
end

torch.traceon(0)


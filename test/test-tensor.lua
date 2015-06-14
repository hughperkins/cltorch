print("running require cltorch...")
require 'cltorch'
print("... require cltorch done")

print(cltorch.getDeviceProperties(cltorch.getDevice()).deviceName)

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

if true then
A = torch.ClTensor{{3,2,4},{9,7,5}}
print('A\n', A)
print('A:sum(2)', A:sum(2))
print('A:sum(1)', A:sum(1))
print('A:max()', A:max())
print('A:min()', A:min())
print('torch.max(A,1)', torch.max(A,1))
print('torch.max(A,2)', torch.max(A,2))
end

if true then
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

if true then
  c = torch.ClTensor{{3,1,6},{2.1,5.2,3.9}}
  c:fill(1.345)
  print('c\n', c)
  c:zero()
  print('c\n', c)
end

if os.getenv('PROTOTYPING') ~= nil then
C = torch.ClTensor{{3,2,4},{9,7,5}}
A = C:float()
print('C\n', C)
D = C:reshape(3,2)
print('D\n', D)
B = A:reshape(3,2)
print('B\n', B)
print(C:t())
end


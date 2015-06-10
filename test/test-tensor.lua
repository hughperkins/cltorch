print("running require clnn...")
require 'clnn'
print("... require clnn done")

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
A = torch.ClTensor{{1,2,-1},
                   {3,4,0}}
B = torch.ClTensor{{0,1},
                   {1,2},
                   {4,5}}
print('A\n', A)
print('B\n', B)
print(torch.mm(A,B))

print(torch.mm(A:float(), B:float()))

C = torch.ClTensor{{0,0},{0,0}}
C:mm(A,B)
print(C)

print( A * B )
C:fill(1.345)
print('C\n', C)

s = torch.LongStorage{3,2}
print('s\n', s)
--C = clnn.ones(s)
--print('C\n', C)
C:zero()
print('C\n', C)

--C:reshape({4,1})
--print('C\n', C)

v1 = torch.ClTensor{3,5,1}
v2 = torch.ClTensor{2,4,8}
print(v1 * v2)

fv1 = torch.FloatTensor{3,5,1}
fv2 = torch.FloatTensor{2,4,8}
print(fv1*fv2)



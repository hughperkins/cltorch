-- unit tests for ClTensor class

luaunit = require('luaunit')
require 'cltorch'

function torch.Tensor.__eq(self, b)
--  print('======= eq begin ====')
--  print('self', self)
  diff = torch.ne(self, b)
--  print('diff', diff)
  sum = torch.sum(diff)
--  print('sum', sum)
  if sum == 0 then
--    print('======= eq end TRUE ====')
    return true
  else
    print('left\n', self)
    print('right\n', b)
    print('diff\n', self - b)
--    print('======= eq end FALSE ====')
    return false
  end
end

function torch.FloatTensor.__eq(self, b)
--  print('======= eq begin ====')
--  print('self', self)
  diff = self - b
--  print('diff1\n', diff)
  diff = torch.abs(diff) - 0.0001
  diff = torch.gt(diff, 0)
--  print('diff', diff)
  sum = torch.sum(diff)
--  print('sum', sum)
  if sum == 0 then
--    print('======= eq end TRUE ====')
    return true
  else
    print('left\n', self)
    print('right\n', b)
    print('diff\n', self - b)
--    print('======= eq end FALSE ====')
    return false
  end
end

function torch.DoubleTensor.__eq(self, b)
--  print('======= eq begin ====')
--  print('self', self)
  diff = self - b
--  print('diff1\n', diff)
  diff = torch.abs(diff) - 0.0001
  diff = torch.gt(diff, 0)
--  print('diff', diff)
  sum = torch.sum(diff)
--  print('sum', sum)
  if sum == 0 then
--    print('======= eq end TRUE ====')
    return true
  else
    print('left\n', self)
    print('right\n', b)
    print('diff\n', self - b)
--    print('======= eq end FALSE ====')
    return false
  end
end

function torch.ClTensor.__eq(self, b)
  print('self', self)
  diff = torch.ne(self, b)
  print('diff', diff)
  sum = torch.sum(diff)
  print('sum', sum)
  if sum == 0 then
    return true
  else
    return false
  end
end

function test_basic()
  c = torch.ClTensor{7,4,5}
  luaunit.assertEquals(' 7\n 4\n 5\n[torch.ClTensor of size 3]\n', tostring(c))

  a = c:float()
  luaunit.assertEquals(' 7\n 4\n 5\n[torch.FloatTensor of size 3]\n', tostring(a))

  c = a:cl()
  luaunit.assertEquals(' 7\n 4\n 5\n[torch.ClTensor of size 3]\n', tostring(c))

  c = torch.ClTensor(3,2)
  luaunit.assertEquals('\n 3\n 2\n[torch.LongStorage of size 2]\n', tostring(c:size()))

  c = torch.ClTensor{{3,1,6},{2.1,5.2,3.9}}
  luaunit.assertEquals(' 3.0000  1.0000  6.0000\n 2.1000  5.2000  3.9000\n[torch.ClTensor of size 2x3]\n', tostring(c))

  d = torch.ClTensor(2,3)
  d:copy(c)
  luaunit.assertEquals(' 3.0000  1.0000  6.0000\n 2.1000  5.2000  3.9000\n[torch.ClTensor of size 2x3]\n', tostring(d))

  c[1][2] = 2.123
  luaunit.assertEquals(' 3.0000  2.1230  6.0000\n 2.1000  5.2000  3.9000\n[torch.ClTensor of size 2x3]\n', tostring(c))
end

function test_clone()
  c = torch.ClTensor{{7,4,5},{3,1,4}}
  luaunit.assertEquals(tostring(c), ' 7  4  5\n 3  1  4\n[torch.ClTensor of size 2x3]\n')
  d = c:clone()  
  luaunit.assertEquals(tostring(d), ' 7  4  5\n 3  1  4\n[torch.ClTensor of size 2x3]\n')
  d[1][2] = 9
  -- should only change d, not c
  luaunit.assertEquals(tostring(c), ' 7  4  5\n 3  1  4\n[torch.ClTensor of size 2x3]\n')
  luaunit.assertEquals(tostring(d), ' 7  9  5\n 3  1  4\n[torch.ClTensor of size 2x3]\n')
end

function test_equals()
  c = torch.ClTensor{{3,5,-2},{2.1,2.2,3.9}}
  d = torch.ClTensor{{3,5,-2},{2.1,2.2,3.9}}
  e = torch.ClTensor{{3,5,-2},{2.1,2.4,3.9}}
  luaunit.assertEquals(c, d)
  luaunit.assertNotEquals(c, e)

  c = c:float()
  d = d:float()
  e = e:float()
  luaunit.assertEquals(c, d)
  luaunit.assertNotEquals(c, e)

  c = torch.Tensor{{3,5,-2},{2.1,2.2,3.9}}
  d = torch.Tensor{{3,5,-2},{2.1,2.2,3.9}}
  e = torch.Tensor{{3,5,-2},{2.1,2.4,3.9}}
  luaunit.assertEquals(c, d)
  luaunit.assertNotEquals(c, e)
end

function test_perelement()
  c = torch.ClTensor{{4,  2,  -1},
                     {3.1,1.2, 4.9}}
  d = torch.ClTensor{{3,  5,  -2},
                     {2.1,2.2, 3.9}}
  a = c:float()
  b = d:float()
  c:add(d)
  a:add(b)
  luaunit.assertEquals(a, c:float())

  c = torch.ClTensor{{4,  2,  -1},
                     {3.1,1.2, 4.9}}
  d = torch.ClTensor{{3,  5,  -2},
                     {2.1,2.2, 3.9}}
  a = c:float()
  b = d:float()
  a:cmul(b)
  c:cmul(d)
  luaunit.assertEquals(a, c:float())

  for _, op in ipairs({'lt', 'gt',
       'le', 'ge', 'eq', 'ne'}) do
    c = torch.ClTensor{{4,  2,  -1},
                       {3.1,1.2, 4.9}}
    d = torch.ClTensor{{3,  5,  -2},
                       {2.1,2.2, 3.9}}
    a = c:float()
    b = d:float()
    print(op)
    loadstring('a:' .. op .. '(b)')()
    loadstring('c:' .. op .. '(d)')()
    luaunit.assertEquals(a:float(), c:float())
    print('   ... ok')
  end

  for _, op in ipairs({'add', 'cmul', 'cdiv', 'cpow', 'cdiv'}) do
    c = torch.ClTensor{{4,  2,  -1},
                       {3.1,1.2, 4.9}}
    d = torch.ClTensor{{3,  5,  -2},
                       {2.1,2.2, 3.9}}
    a = c:float()
    b = d:float()
    print(op)
    loadstring('a:' .. op .. '(b)')()
    loadstring('c:' .. op .. '(d)')()
    luaunit.assertEquals(a, c:float())
    print('   ... ok')
  end

  for _, op in ipairs({'lt', 'gt',
       'le', 'ge', 'eq', 'ne'}) do
    c = torch.ClTensor{{4,  2,  -1},
                       {3.1,1.2, 4.9}}
    d = torch.ClTensor{{3,  5,  -2},
                       {2.1,2.2, 3.9}}
    a = c:float()
    b = d:float()
    print(op)
    loadstring('a = torch.' .. op .. '(a, b)')()
    loadstring('c = torch.' .. op .. '(c, d)')()
    luaunit.assertEquals(a:float(), c:float())
    print('   ... ok')
  end

  for _, op in ipairs({'add', 'cmul', 'cdiv', 'cpow', 'cdiv'}) do
    c = torch.ClTensor{{4,  2,  -1},
                       {3.1,1.2, 4.9}}
    d = torch.ClTensor{{3,  5,  -2},
                       {2.1,2.2, 3.9}}
    a = c:float()
    b = d:float()
    print(op)
    loadstring('a = torch.' .. op .. '(a, b)')()
    loadstring('c = torch.' .. op .. '(c, d)')()
    luaunit.assertEquals(a, c:float())
    print('   ... ok')
  end

  for _, op in ipairs({'+', '-'}) do
    c = torch.ClTensor{{4,  2,  -1},
                       {3.1,1.2, 4.9}}
    d = torch.ClTensor{{3,  5,  -2},
                       {2.1,2.2, 3.9}}
    a = c:float()
    b = d:float()
    print(op)
    loadstring('a = a ' .. op .. ' b')()
    loadstring('c = c ' .. op .. ' d')()
    luaunit.assertEquals(a, c:float())
    print('   ... ok')
  end

  for _,name in ipairs({'abs', 'sqrt', 'log','exp', 'cos', 
     'acos', 'sin', 'asin', 'atan', 'tanh', 'ceil', 'floor', 
     'abs', 'round'}) do
    c = torch.ClTensor{{4,  2,  -1},
                       {3.1,1.2, 4.9}}
    a = c:float()
    print(name)
    loadstring('c:' .. name .. '()')()
    loadstring('a:' .. name .. '()')()
    luaunit.assertEquals(a, c:float())
    print('   ... ok')
  end

  for _,name in ipairs({'lt','le','gt','ge','ne','eq'}) do
    print(name)
    c = torch.ClTensor{{4,  2,  -1},
                       {3.1,1.2, 4.9}}
    d = torch.ClTensor{{3,  5,  -2},
                       {2.1,2.2, 3.9}}
    a = c:float()
    b = d:float()
    print(loadstring('c = torch.' .. name .. '(c,d)')())
    print(loadstring('a = torch.' .. name .. '(a,b)')())
    luaunit.assertEquals(a:float(), c:float())
    print('   ... ok')
  end

  for _,name in ipairs({'add', 'mul', 'div', 'pow',
      'lt', 'le', 'gt', 'ge', 'ne', 'eq'}) do
    c = torch.ClTensor{{4,  2,  -1},
                       {3.1,1.2, 4.9}}
    a = c:float()
    print(name)
    print(loadstring('c = torch.' .. name .. '(c, 3.4)')())
    print(loadstring('a = torch.' .. name .. '(a, 3.4)')())
    luaunit.assertEquals(a:float(), c:float())
    print('   ... ok')    
  end

  for _,name in ipairs({'+', '/', '*', '-'}) do
    c = torch.ClTensor{{4,  2,  -1},
                       {3.1,1.2, 4.9}}
    a = c:float()
    print(name)
    print(loadstring('c = c ' .. name .. ' 3.4')())
    print(loadstring('a = a ' .. name .. ' 3.4')())
    luaunit.assertEquals(a, c:float())
    print('   ... ok')    
  end

  for _,name in ipairs({'lt','le','gt','ge','ne','eq'}) do
    print('name', name)
    print(loadstring('return c:' .. name .. '(5)')())
  end

  c = torch.ClTensor{{4,  2,  -1},
                     {3.1,1.2, 4.9}}
  a = c:float()
  a = torch.clamp(a, 1.5, 4.5)
  c = torch.clamp(c, 1.5, 4.5)
  luaunit.assertEquals(a, c:float())

  c = torch.ClTensor{{4,  2,  -1},
                     {3.1,1.2, 4.9}}
  a = c:float()
  c = -c
  a = -a
  luaunit.assertEquals(a, c:float())
end

function test_blas()
  C = torch.ClTensor{{1,2,-1},
                   {3,4,0}}
  D = torch.ClTensor{{0,1},
                     {1,2},
                     {4,5}}
  A = C:float()
  B = D:float()
  A = torch.mm(A,B)
  C = torch.mm(C,D)
  print('A\n', A)
  print('C\n', C)
  luaunit.assertEquals(A, C:float())

  C = torch.ClTensor{{1,2,-1},
                   {3,4,0}}
  D = torch.ClTensor{{0,1},
                     {1,2},
                     {4,5}}
  A = C:float()
  B = D:float()
  A = A * B
  C = C * D
  print('A\n', A)
  print('C\n', C)
  luaunit.assertEquals(A, C:float())

  c = torch.ClTensor{3,5,1}
  d = torch.ClTensor{2,4,8}
  a = c:float()
  b = d:float()
  a = torch.dot(a,b)
  c = torch.dot(c,d)
  luaunit.assertEquals(a, c)

  c = torch.ClTensor{3,5,1}
  d = torch.ClTensor{2,4,8}
  a = c:float()
  b = d:float()
  a = a * b
  c = c * d
  luaunit.assertEquals(a, c)

  C = torch.ClTensor{{3,1,7},{3,2,4},{8,5,3}}
  d = torch.ClTensor{3,2,5}
  e = torch.ClTensor{3,1,2}
  
  luaunit.assertEquals((torch.addr(C,d,e)):float(), torch.addr(C:float(), d:float(), e:float()))
end

function test_fills()
  C = torch.ClTensor(3,2)
  A = torch.FloatTensor(3,2)
  C:fill(1.345)
  A:fill(1.345)
  luaunit.assertEquals(A, C:float())

  C = torch.ClTensor(3,2)
  A = torch.FloatTensor(3,2)
  C:fill(1.345)
  A:fill(1.345)
  C:zero()
  A:zero()
  luaunit.assertEquals(A, C:float())
  
  A = torch.FloatTensor.zeros(torch.FloatTensor.new(), 3, 5)
  C = torch.ClTensor.zeros(torch.ClTensor.new(), 3, 5)
  luaunit.assertEquals(A, C:float())

  A = torch.FloatTensor.ones(torch.FloatTensor.new(), 3, 5)
  C = torch.ClTensor.ones(torch.ClTensor.new(), 3, 5)
  luaunit.assertEquals(A, C:float())

end

function test_matrixwide()
  C = torch.ClTensor{{3,2,4},{9,7,5}}
  A = C:float()
  luaunit.assertEquals(A:max(), C:max())
  luaunit.assertEquals(A:min(), C:min())
  luaunit.assertEquals(A:sum(), C:sum())
  luaunit.assertEquals(A:sum(1), C:sum(1):float())
  luaunit.assertEquals(A:sum(2), C:sum(2):float())
end

function test_reshape()
  C = torch.ClTensor{{3,2,4},{9,7,5}}
  A = C:float()
  D = C:reshape(3,2)
  B = A:reshape(3,2)
  luaunit.assertEquals(B, D:float())
end

function test_intpower()
  C = torch.ClTensor{{3.3,-2.2,0},{9,7,5}}
  Cpow = torch.pow(C,2)
  A = C:float()
  Apow = torch.pow(A,2)
  luaunit.assertEquals(Apow, Cpow:float())
end

function test_powerofneg()
  C = torch.ClTensor{{3.3,-2.2,0},{9,7,5}}
  Cpow = torch.pow(C,2.4)
  A = C:float()
  Apow = torch.pow(A,2.4)
  luaunit.assertEquals(Apow, Cpow:float())
end

function test_add()
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform()
  local B = torch.Tensor(s):uniform()
  luaunit.assertEquals(torch.add(A,B), torch.add(A:clone():cl(), B:clone():cl()):double())
  luaunit.assertEquals(A + B, (A:clone():cl() + B:clone():cl() ):double())
  luaunit.assertEquals(A:clone():add(B), (A:clone():cl():add(B:clone():cl())):double())
end

function test_cmul()
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform()
  local B = torch.Tensor(s):uniform()
  luaunit.assertEquals(torch.cmul(A,B), torch.cmul(A:clone():cl(), B:clone():cl()):double())
  luaunit.assertEquals(A:clone():cmul(B), (A:clone():cl():cmul(B:clone():cl())):double())
end

-- this function doesnt exist in base torch
function test_neg()
  -- no neg for Tensors, only for clTensor, but we can use '-' to 
  -- compare
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform()
  local negA = - A:clone()
  local negAcl1 = - A:clone():cl()
  local negAcl2 = A:clone():cl():neg()
  luaunit.assertEquals(negA, negAcl1:double())
  luaunit.assertEquals(negA, negAcl2:double())
end

-- this function doesnt exist in base torch
function test_sub()
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform()
  local B = torch.Tensor(s):uniform()
  AsubB = A - B
  AsubBcl = A:clone():cl() - B:clone():cl()
  AsubBcl2 = A:clone():cl():sub(B:clone():cl())
--  AsubBcl3 = torch.sub(A:clone():cl(), B:clone():cl())
  luaunit.assertEquals(AsubB, AsubBcl:double())
  luaunit.assertEquals(AsubB, AsubBcl2:double())
--  luaunit.assertEquals(AsubB, AsubBcl3:double())
end

function test_apply()
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform()
  local Aapply = A:clone():apply(function(x) return math.sqrt(x+3) end)
  local Aapplycl = A:clone():cl():apply("*out = sqrt(*out + 3)")
  luaunit.assertEquals(Aapply, Aapplycl:double())
end

function test_map()
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform()
  local B = torch.Tensor(s):uniform()
  local AmapB = A:clone():map(B, function(x, y) return math.sqrt(x*x + y*y + 3) end)
  local AmapBcl = A:clone():cl():map(B:clone():cl(), 
    "*out = sqrt(*out * *out + *in1 * *in1 + 3)")
  luaunit.assertEquals(AmapB, AmapBcl:double())
end

function test_map2()
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform()
  local B = torch.Tensor(s):uniform()
  local C = torch.Tensor(s):uniform()
  local Amap2BC = A:clone():map2(B, C, 
    function(x, y, z) return math.sqrt(x*x + y*y + z + 3) end)
  local Amap2BCcl = A:clone():cl():map2(
    B:clone():cl(),
    C:clone():cl(), 
    "*out = sqrt(*out * *out + *in1 * *in1 + *in2 + 3)")
  luaunit.assertEquals(Amap2BC, Amap2BCcl:double())
end

function test_reduceAll()
  -- test on a large tensor, that needs two-pass :-)
  A = torch.Tensor(28*28*1280,10):uniform()
  Asum = torch.sum(A)
  Aclsum = torch.sum(A:clone():cl())
  diff = Asum - Aclsum
  if diff < 0 then
    diff = - diff
  end
  luaunit.assertTrue(diff < 0.5)

  -- now test on a single pass
  A = torch.Tensor(50,40):uniform()
  Asum = torch.sum(A)
  Aclsum = torch.sum(A:clone():cl())
  diff = Asum - Aclsum
  if diff < 0 then
    diff = - diff
  end
  luaunit.assertTrue(diff < 0.1)
end

os.exit( luaunit.LuaUnit.run() )



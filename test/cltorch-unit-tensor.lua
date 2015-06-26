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
  diff = torch.abs(diff) - 0.0002
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

function torch.LongTensor.__eq(self, b)
  diff = self - b
  diff = torch.abs(diff)
  sum = torch.sum(diff)
  if sum == 0 then
    return true
  else
    print('left\n', self)
    print('right\n', b)
    print('diff\n', self - b)
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

-- not allowed any more (at least: not for now, unless we can find an efficient
-- way of implementing this?)
--  c[1][2] = 2.123
--  luaunit.assertEquals(' 3.0000  2.1230  6.0000\n 2.1000  5.2000  3.9000\n[torch.ClTensor of size 2x3]\n', tostring(c))
end

function test_clone()
  c = torch.ClTensor{{7,4,5},{3,1,4}}
  luaunit.assertEquals(tostring(c), ' 7  4  5\n 3  1  4\n[torch.ClTensor of size 2x3]\n')
  d = c:clone()  
  luaunit.assertEquals(tostring(d), ' 7  4  5\n 3  1  4\n[torch.ClTensor of size 2x3]\n')
  --d[1][2] = 9
  d:fill(3)
  -- should only change d, not c
  luaunit.assertEquals(tostring(c), ' 7  4  5\n 3  1  4\n[torch.ClTensor of size 2x3]\n')
  luaunit.assertEquals(tostring(d), ' 3  3  3\n 3  3  3\n[torch.ClTensor of size 2x3]\n')
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
  local A = torch.Tensor(s):uniform() - 0.5
  local B = torch.Tensor(s):uniform() - 0.5
  luaunit.assertEquals(torch.add(A,B), torch.add(A:clone():cl(), B:clone():cl()):double())
  luaunit.assertEquals(A + B, (A:clone():cl() + B:clone():cl() ):double())
  luaunit.assertEquals(A:clone():add(B), (A:clone():cl():add(B:clone():cl())):double())
end

function test_cmul()
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform() - 0.5
  local B = torch.Tensor(s):uniform() - 0.5
  luaunit.assertEquals(torch.cmul(A,B), torch.cmul(A:clone():cl(), B:clone():cl()):double())
  luaunit.assertEquals(A:clone():cmul(B), (A:clone():cl():cmul(B:clone():cl())):double())
end

function test_addcmul()
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform() - 0.5
  local B = torch.Tensor(s):uniform() - 0.5
  local C = torch.Tensor(s):uniform() - 0.5
  luaunit.assertEquals(torch.addcmul(A,1.234,B,C), torch.addcmul(A:clone():cl(), 1.234, B:clone():cl(), C:clone():cl()):double())
  luaunit.assertEquals(A:clone():addcmul(1.234,B,C), (A:clone():cl():addcmul(1.234, B:clone():cl(),C:clone():cl())):double())
end

function test_addcdiv()
  local s = torch.LongStorage{20,50}
  local A = torch.Tensor(s):uniform() - 0.5
  local B = torch.Tensor(s):uniform() - 0.5
  local C = torch.Tensor(s):uniform() - 0.5
  luaunit.assertEquals(torch.addcdiv(A,1.234,B,C), torch.addcdiv(A:clone():cl(), 1.234, B:clone():cl(), C:clone():cl()):double())
  luaunit.assertEquals(A:clone():addcdiv(1.234,B,C), (A:clone():cl():addcdiv(1.234, B:clone():cl(),C:clone():cl())):double())
end

-- this function doesnt exist in base torch
function test_neg()
  -- no neg for Tensors, only for clTensor, but we can use '-' to 
  -- compare
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform() - 0.5
  local negA = - A:clone()
  local negAcl1 = - A:clone():cl()
  local negAcl2 = A:clone():cl():neg()
  luaunit.assertEquals(negA, negAcl1:double())
  luaunit.assertEquals(negA, negAcl2:double())
end

-- this function doesnt exist in base torch
function test_sub()
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform() - 0.5
  local B = torch.Tensor(s):uniform() - 0.5
  AsubB = A - B
  AsubBcl = A:clone():cl() - B:clone():cl()
  AsubBcl2 = A:clone():cl():csub(B:clone():cl())
--  AsubBcl3 = torch.sub(A:clone():cl(), B:clone():cl())
  luaunit.assertEquals(AsubB, AsubBcl:double())
  luaunit.assertEquals(AsubB, AsubBcl2:double())
--  luaunit.assertEquals(AsubB, AsubBcl3:double())
end

function test_apply()
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform() - 0.5
  local Aapply = A:clone():apply(function(x) return math.sqrt(x+3) end)
  local Aapplycl = A:clone():cl():apply("*out = sqrt(*out + 3)")
  luaunit.assertEquals(Aapply, Aapplycl:double())

  local Aapplycl_x = A:clone():cl():apply("x = sqrt(x + 3)")
  luaunit.assertEquals(Aapply, Aapplycl_x:double())
end

function test_map()
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform() - 0.5
  local B = torch.Tensor(s):uniform() - 0.5
  local AmapB = A:clone():map(B, function(x, y) return math.sqrt(x*x + y*y + 3) end)
  local AmapBcl = A:clone():cl():map(B:clone():cl(), 
    "*out = sqrt(*out * *out + *in1 * *in1 + 3)")
  luaunit.assertEquals(AmapB, AmapBcl:double())

  local Aapp2Bcl = A:clone():cl():apply2(B:clone():cl(), 
    "*out = sqrt(*out * *out + *in1 * *in1 + 3)")
  luaunit.assertEquals(AmapB, Aapp2Bcl:double())

  local Aapp2Bcl_xy = A:clone():cl():apply2(B:clone():cl(), 
    "x = sqrt(x * x + y * y + 3)")
  luaunit.assertEquals(AmapB, Aapp2Bcl_xy:double())
end

function test_map2()
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform() - 0.5
  local B = torch.Tensor(s):uniform() - 0.5
  local C = torch.Tensor(s):uniform() - 0.5
  local Amap2BC = A:clone():map2(B, C, 
    function(x, y, z) return math.sqrt(x*x + y*y + z + 3) end)
  local Amap2BCcl = A:clone():cl():map2(
    B:clone():cl(),
    C:clone():cl(), 
    "*out = sqrt(*out * *out + *in1 * *in1 + *in2 + 3)")

  local Aapp3BCcl = A:clone():cl():apply3(
    B:clone():cl(),
    C:clone():cl(), 
    "*out = sqrt(*out * *out + *in1 * *in1 + *in2 + 3)")
  luaunit.assertEquals(Amap2BC, Aapp3BCcl:double())

  local Aapp3BCcl_xyz = A:clone():cl():apply3(
    B:clone():cl(),
    C:clone():cl(), 
    "x = sqrt(x * x + y * y + z + 3)")
  luaunit.assertEquals(Amap2BC, Aapp3BCcl_xyz:double())
end

function test_reduceAll()
  -- test on a large tensor, that needs two-pass :-)
  A = torch.Tensor(28*28*1280,10):uniform() - 0.5
  Asum = torch.sum(A)
  Aclsum = torch.sum(A:clone():cl())
  diff = Asum - Aclsum
  if diff < 0 then
    diff = - diff
  end
  luaunit.assertTrue(diff < 1.2)

  -- now test on a single pass
  A = torch.Tensor(50,40):uniform() - 0.5
  Asum = torch.sum(A)
  Aclsum = torch.sum(A:clone():cl())
  diff = Asum - Aclsum
  if diff < 0 then
    diff = - diff
  end
  luaunit.assertTrue(diff < 0.1)
end

function test_prodall()
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform() - 0.5
  local Acl = A:cl()
--  print('allocated A,Acl')
  local Aprodall = torch.prod(A)
  local Aclprodall = torch.prod(Acl)
--  print('done prodall calcs')
  luaunit.assertEquals(Aprodall, Aclprodall)

  Aprodall2 = A:prod()
--  print('calcing...')
  Aclprodall2 = Acl:prod()
--  print('...calced')
  luaunit.assertEquals(Aprodall, Aprodall2)
  luaunit.assertEquals(Aprodall, Aclprodall2)
end

function test_sumall()
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform() - 0.5
  local Acl = A:cl()
--  print('allocated A,Acl')
  local Asumall = torch.sum(A)
  local Aclsumall = torch.sum(Acl)
--  print('done sumall calcs')
  luaunit.assertEquals(torch.FloatTensor{Asumall}, torch.FloatTensor{Aclsumall})

  Aclsumall2 = Acl:sum()
  luaunit.assertEquals(torch.FloatTensor{Asumall}, torch.FloatTensor{Aclsumall2})
end

function test_sumallt()
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform() - 0.5
  local Acl = A:cl()
--  print('allocated A,Acl')
  local Asumall = torch.sum(A:t())
  local Aclsumall = torch.sum(Acl:t())
--  print('done sumall calcs')
  luaunit.assertEquals(torch.FloatTensor{Asumall}, torch.FloatTensor{Aclsumall})

  Aclsumall2 = Acl:sum()
  luaunit.assertEquals(torch.FloatTensor{Asumall}, torch.FloatTensor{Aclsumall2})
end

function test_prod()
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform() - 0.5
  local Acl = A:cl()
  luaunit.assertEquals(A:prod(1), Acl:prod(1):double())  
  luaunit.assertEquals(A:prod(2), Acl:prod(2):double())  
end

function test_sum()
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform() - 0.5
  local Acl = A:cl()
  luaunit.assertEquals(A:sum(1), Acl:sum(1):double())  
  luaunit.assertEquals(A:sum(2), Acl:sum(2):double())  
end

function test_sum_t()
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform() - 0.5
  local Acl = A:cl()
  luaunit.assertEquals(A:t():sum(1), Acl:t():sum(1):double())  
  luaunit.assertEquals(A:t():sum(2), Acl:t():sum(2):double())  
end

function test_sum_t_offset()
  local s = torch.LongStorage{60,50}
  local A = torch.Tensor(s):uniform() - 0.5
  local Acl = A:cl()
  luaunit.assertEquals(A:narrow(1,10,30):t():sum(1), Acl:narrow(1,10,30):t():sum(1):double())  
  luaunit.assertEquals(A:narrow(1,10,30):t():sum(2), Acl:narrow(1,10,30):t():sum(2):double())  
  luaunit.assertEquals(A:narrow(2,10,30):t():sum(1), Acl:narrow(2,10,30):t():sum(1):double())  
  luaunit.assertEquals(A:narrow(2,10,30):t():sum(2), Acl:narrow(2,10,30):t():sum(2):double())  
end

--function test_sum_t_offset()
--  local s = torch.LongStorage{60,50}
--  local A = torch.Tensor(s):uniform() - 0.5
----  A = A:narrow(2, 10, 30)
--  local Acl = A:cl()
--  luaunit.assertEquals(A:t():sum(1), Acl:t():sum(1):double())  
--  luaunit.assertEquals(A:t():sum(2), Acl:t():sum(2):double())  
--end

function test_max1()
  local s = torch.LongStorage{5,2}
  local A = torch.Tensor(s):uniform() - 0.5
  local Acl = A:cl()
  local Amax, Aind = A:max(1)
  local Aclmax, Aclind = Acl:max(1)
--  print('A max', Amax, Aind)
--  print('Acl max', Aclmax, Aclind)
  luaunit.assertEquals(Amax, Aclmax:double())
  luaunit.assertEquals(Aind, Aclind:long())
end

function test_max2()
  local s = torch.LongStorage{5,2}
  local A = torch.Tensor(s):uniform() - 0.5
  local Acl = A:cl()
  local Amax, Aind = A:max(2)
  local Aclmax, Aclind = Acl:max(2)
--  print('A max', Amax, Aind)
--  print('Acl max', Aclmax, Aclind)
  luaunit.assertEquals(Amax, Aclmax:double())
  luaunit.assertEquals(Aind, Aclind:long())
end

function test_min1()
  local s = torch.LongStorage{5,2}
  local A = torch.Tensor(s):uniform() - 0.5
  local Acl = A:cl()
  local Amax, Aind = A:min(1)
  local Aclmax, Aclind = Acl:min(1)
--  print('A max', Amax, Aind)
--  print('Acl max', Aclmax, Aclind)
  luaunit.assertEquals(Amax, Aclmax:double())
  luaunit.assertEquals(Aind, Aclind:long())
end

function test_min2()
  local s = torch.LongStorage{5,2}
  local A = torch.Tensor(s):uniform() - 0.5
  local Acl = A:cl()
  local Amax, Aind = A:min(2)
  local Aclmax, Aclind = Acl:min(2)
--  print('A max', Amax, Aind)
--  print('Acl max', Aclmax, Aclind)
  luaunit.assertEquals(Amax, Aclmax:double())
  luaunit.assertEquals(Aind, Aclind:long())
end

function test_indexfill()
  x = torch.Tensor(60,50):uniform() - 0.5
  selector = torch.LongTensor{4,7,12,19,35}
  value = -7
  xfill = x:clone()
  xfill:indexFill(2, selector, value)
--  print('xfill\n', xfill)

  xfillcl = x:clone():cl()
  xfillcl:indexFill(2, selector, value)
--  print('xfillcl\n', xfillcl)

  luaunit.assertEquals(xfill, xfillcl:double())
end

function test_indexcopy()
  x = torch.Tensor(60,50):uniform() - 0.5
  z = torch.Tensor(60,2)
  z:select(2,1):fill(-1)
  z:select(2,2):fill(-2)
  xafter = x:clone()
  xafter:indexCopy(2,torch.LongTensor{5,1}, z)

  xaftercl = x:cl()
  xaftercl:indexCopy(2,torch.LongTensor{5,1}, z:cl())
  luaunit.assertEquals(xafter, xaftercl:double())
end

function test_indexselect()
  local x = torch.Tensor(60,50):uniform() - 0.5

  xcopy = x:clone()
  xcopy:select(1, 2):fill(2) -- select row 2 and fill up
  xcopy:select(2, 5):fill(5) -- select column 5 and fill up

  xcopycl = x:cl()
  xcopycl:select(1, 2):fill(2) -- select row 2 and fill up
  xcopycl:select(2, 5):fill(5) -- select column 5 and fill up
  
  luaunit.assertEquals(xcopy, xcopycl:double())
end

function test_cumsum()
  x = torch.Tensor(50,60):uniform() - 0.5
  luaunit.assertEquals(torch.cumsum(x), torch.cumsum(x:cl()):double())
  luaunit.assertEquals(torch.cumsum(x, 1), torch.cumsum(x:cl(), 1):double())
  luaunit.assertEquals(torch.cumsum(x, 2), torch.cumsum(x:cl(), 2):double())
end

function test_cumprod()
  x = torch.Tensor(50,60):uniform() - 0.5
  luaunit.assertEquals(torch.cumprod(x), torch.cumprod(x:cl()):double())
  luaunit.assertEquals(torch.cumprod(x, 1), torch.cumprod(x:cl(), 1):double())
  luaunit.assertEquals(torch.cumprod(x, 2), torch.cumprod(x:cl(), 2):double())
end

function test_gather()
  a = torch.Tensor(60,4)
  a:copy(torch.range(1,a:nElement()))
  acl = a:clone():cl()
  idx = torch.LongTensor({{2,1,3,1}})
  idxcl = idx:clone():cl()

  luaunit.assertEquals(a:gather(1, idx), acl:gather(1, idxcl):double())
  luaunit.assertEquals(torch.gather(a, 1, idx), torch.gather(acl, 1, idxcl):double())
end

function test_gather_t()
  a = torch.Tensor(60,4)
  a:copy(torch.range(1,a:nElement()))
  acl = a:clone():cl()

  idx = torch.LongTensor({{2,1,3,1}})
  idxcl = idx:clone():cl()

  a = a:t()
  acl = acl:t()
  idx = idx:t()
  idxcl = idxcl:t()

  luaunit.assertEquals(a:gather(2, idx), acl:gather(2, idxcl):double())
  luaunit.assertEquals(torch.gather(a, 2, idx), torch.gather(acl, 2, idxcl):double())
end

function test_gather_narrowed()
  a = torch.Tensor(60,4)
  a:copy(torch.range(1,a:nElement()))
  acl = a:clone():cl()

  idx = torch.LongTensor({{2,1,3,1}})
  idxcl = idx:clone():cl()

  a = a:narrow(1, 20, 20)
  acl = acl:narrow(1, 20, 20)

  cltorch.setTrace(1)
  luaunit.assertEquals(a:gather(1, idx), acl:gather(1, idxcl):double())
  luaunit.assertEquals(torch.gather(a, 1, idx), torch.gather(acl, 1, idxcl):double())
  cltorch.setTrace(0)
end

function test_scatter()
  x = torch.Tensor(2,5)
  x = x:copy(torch.range(1, x:nElement()))
  y = torch.zeros(3,5)
  idx = torch.LongTensor{{1,2,3,1,1},{3,1,1,2,3}}

  z = y:clone():scatter(1, idx:clone(), x:clone())
  zcl = y:clone():cl():scatter(1, idx:clone():cl(), x:clone():cl())

  luaunit.assertEquals(z, zcl:double())
end

function test_scatterFill()
  y = torch.zeros(3,5)
  idx = torch.LongTensor{{1,2,3,1,1},{3,1,1,2,3}}

  z = y:clone():scatter(1, idx:clone(), 3.4567)
  zcl = y:clone():cl():scatter(1, idx:clone():cl(), 3.4567)

  luaunit.assertEquals(z, zcl:double())
end
local function _run()
  --cltorch.setTrace(1)
  luaunit.LuaUnit.run()
  -- cltorch.setTrace(0)
end

--cltorch.setTrace(1)
luaunit.LuaUnit.run()
cltorch.setTrace(0)
-- os.exit(_run())
--os.exit(luaunit.LuaUnit.run())


-- unit tests for ClTensor class

require 'string'

local runtests = false
if not cltorch then
    print('requiring cltorch')
    require 'cltorch'
    runtests = true
end

if not cltorch.tests then
   cltorch.tests = {}
end

cltorch.tests.tensor = {}

local function assertStrContains(target, value )
   local res = string.find(target, value)
   if res == nil then
      print('assertStrContains fail: [' .. string.gsub(target, '\n', '\\n\n') .. '] not contains [' .. string.gsub(value, '\n', '\\n\n') .. ']')
      tester:assert(string.find(target, value) ~= nil)
   end
end

-- probalby not ideal to modify the original Tensor classes, but anyway...

function torch.Tensor.__eq(self, b)
--   print('======= eq begin ====')
--   print('self', self)
   diff = torch.ne(self, b)
--   print('diff', diff)
   sum = torch.sum(diff)
--   print('sum', sum)
   if sum == 0 then
--      print('======= eq end TRUE ====')
      return true
   else
      print('left\n', self)
      print('right\n', b)
--      print('diff\n', self - b)
--      print('======= eq end FALSE ====')
      return false
   end
end

function torch.FloatTensor.__eq(self, b)
--   print('======= eq begin ====')
--   print('self', self)
   diff = self - b
--   print('diff1\n', diff)
   diff = torch.abs(diff) - 0.0002
   diff = torch.gt(diff, 0)
--   print('diff', diff)
   sum = torch.sum(diff)
--   print('sum', sum)
   if sum == 0 then
--      print('======= eq end TRUE ====')
      return true
   else
      print('left\n', self)
      print('right\n', b)
--      print('diff\n', self - b)
--      print('======= eq end FALSE ====')
      return false
   end
end

function torch.DoubleTensor.__eq(self, b)
--   print('======= eq begin ====')
--   print('self', self)
   diff = self - b
--   print('diff1\n', diff)
   diff = torch.abs(diff) - 0.0001
   diff = torch.gt(diff, 0)
--   print('diff', diff)
   sum = torch.sum(diff)
--   print('sum', sum)
   if sum == 0 then
--      print('======= eq end TRUE ====')
      return true
   else
      print('left\n', self)
      print('right\n', b)
--      print('diff\n', self - b)
--      print('======= eq end FALSE ====')
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
--      print('diff\n', self - b)
      return false
   end
end

function torch.LongStorage.__eq(self, b)
--   print('LongStorage.__eq()')
   if self:size() ~= b:size() then
      return false
   end
--   tester:asserteq(self:size(), b:size())
   for i=1,self:size() do
      if self[i] ~= b[i] then
         return false
      end
--      tester:asserteq(self[i], b[i])
   end
   return true
end

function torch.ClTensor.__eq(self, b)
--   print('self', self)
   diff = torch.ne(self, b)
--   print('diff', diff)
   sum = torch.sum(diff)
--   print('sum', sum)
   if sum == 0 then
      return true
   else
      return false
   end
end

function cltorch.tests.tensor.test_basic()
   c = torch.ClTensor{7,4,5}
   tester:asserteq(' 7\n 4\n 5\n[torch.ClTensor of size 3]\n', tostring(c))

   a = c:float()
   tester:asserteq(' 7\n 4\n 5\n[torch.FloatTensor of size 3]\n', tostring(a))

   c = a:cl()
   tester:asserteq(' 7\n 4\n 5\n[torch.ClTensor of size 3]\n', tostring(c))

   c = torch.ClTensor(3,2)
   tester:asserteq('\n 3\n 2\n[torch.LongStorage of size 2]\n', tostring(c:size()))

   c = torch.ClTensor{{3,1,6},{2.1,5.2,3.9}}
   tester:asserteq(' 3.0000  1.0000  6.0000\n 2.1000  5.2000  3.9000\n[torch.ClTensor of size 2x3]\n', tostring(c))

   d = torch.ClTensor(2,3)
   d:copy(c)
   tester:asserteq(' 3.0000  1.0000  6.0000\n 2.1000  5.2000  3.9000\n[torch.ClTensor of size 2x3]\n', tostring(d))

-- not allowed any more (at least: not for now, unless we can find an efficient
-- way of implementing this?)
--   c[1][2] = 2.123
--   tester:asserteq(' 3.0000   2.1230   6.0000\n 2.1000   5.2000   3.9000\n[torch.ClTensor of size 2x3]\n', tostring(c))
end

function cltorch.tests.tensor.test_clone()
   c = torch.ClTensor{{7,4,5},{3,1,4}}
   tester:asserteq(tostring(c), ' 7  4  5\n 3  1  4\n[torch.ClTensor of size 2x3]\n')
   d = c:clone()
   tester:asserteq(tostring(d), ' 7  4  5\n 3  1  4\n[torch.ClTensor of size 2x3]\n')
   --d[1][2] = 9
   d:fill(3)
   -- should only change d, not c
   tester:asserteq(tostring(c), ' 7  4  5\n 3  1  4\n[torch.ClTensor of size 2x3]\n')
   tester:asserteq(tostring(d), ' 3  3  3\n 3  3  3\n[torch.ClTensor of size 2x3]\n')
end

function cltorch.tests.tensor.test_equals()
   c = torch.ClTensor{{3,5,-2},{2.1,2.2,3.9}}
   d = torch.ClTensor{{3,5,-2},{2.1,2.2,3.9}}
   e = torch.ClTensor{{3,5,-2},{2.1,2.4,3.9}}
   tester:asserteq(c, d)
   tester:assertne(c, e)

   c = c:float()
   d = d:float()
   e = e:float()
   tester:asserteq(c, d)
   tester:assertne(c, e)

   c = torch.Tensor{{3,5,-2},{2.1,2.2,3.9}}
   d = torch.Tensor{{3,5,-2},{2.1,2.2,3.9}}
   e = torch.Tensor{{3,5,-2},{2.1,2.4,3.9}}
   tester:asserteq(c, d)
   tester:assertne(c, e)
end

for _, op in ipairs({'lt', 'gt',
       'le', 'ge', 'eq', 'ne'}) do
   cltorch.tests.tensor['inplace_' .. op] = function()
      c = torch.ClTensor{{4,    2,  -1},
                         {3.1,1.2, 4.9}}
      d = torch.ClTensor{{3,    5,  -2},
                         {2.1,2.2, 3.9}}
      a = c:float()
      b = d:float()
      loadstring('a:' .. op .. '(b)')()
      loadstring('c:' .. op .. '(d)')()
      tester:asserteq(a:float(), c:float())
   end
end

for _,name in ipairs({'abs', 'sqrt', 'log','exp', 'cos', 
    'acos', 'sin', 'asin', 'atan', 'tanh', 'ceil', 'floor', 
    'abs', 'round'}) do
   cltorch.tests.tensor['inplace_' .. name] = function()
      c = torch.ClTensor{{4,    2,  -1},
                         {3.1,1.2, 4.9}}
      a = c:float()
      loadstring('c:' .. name .. '()')()
      loadstring('a:' .. name .. '()')()
      tester:asserteq(a, c:float())
   end
end

for _, op in ipairs({'add', 'cmul', 'cdiv', 'cpow', 'cdiv'}) do
   cltorch.tests.tensor['inplace_' .. op] = function()
      c = torch.ClTensor{{4,   2,   -1},
                         {0.8,1.2, 1.9}}
      d = torch.ClTensor{{3,   5,   -2},
                         {2.1,2.2, 0.9}}
      a = c:float()
      b = d:float()
      loadstring('a:' .. op .. '(b)')()
      loadstring('c:' .. op .. '(d)')()
      tester:asserteq(a, c:float())
   end
end

for _, op in ipairs({'lt', 'gt',
          'le', 'ge', 'eq', 'ne'}) do
   cltorch.tests.tensor['outplace_' .. op] = function()
      c = torch.ClTensor{{4,   2,   -1},
                         {3.1,1.2, 4.9}}
      d = torch.ClTensor{{3,   5,   -2},
                         {2.1,2.2, 3.9}}
      a = c:float()
      b = d:float()
      loadstring('a = torch.' .. op .. '(a, b)')()
      loadstring('c = torch.' .. op .. '(c, d)')()
      tester:asserteq(a:float(), c:float())
   end
end

for _, op in ipairs({'add', 'cmul', 'cdiv', 'cpow', 'cdiv'}) do
   cltorch.tests.tensor['outplace_' .. op] = function()
      c = torch.ClTensor{{4,   2,   -1},
                         {0.8,1.2, 1.9}}
      d = torch.ClTensor{{3,   5,   -2},
                         {2.1,2.2, 0.9}}
      a = c:float()
      b = d:float()
      loadstring('a = torch.' .. op .. '(a, b)')()
      loadstring('c = torch.' .. op .. '(c, d)')()
      tester:asserteq(a, c:float())
   end
end

for _,name in ipairs({'lt','le','gt','ge','ne','eq'}) do
   cltorch.tests.tensor['outplace_' .. name] = function()
      c = torch.ClTensor{{4,   2,   -1},
                          {3.1,1.2, 4.9}}
      d = torch.ClTensor{{3,   5,   -2},
                          {2.1,2.2, 3.9}}
      a = c:float()
      b = d:float()
      print(loadstring('c = torch.' .. name .. '(c,d)')())
      print(loadstring('a = torch.' .. name .. '(a,b)')())
      tester:asserteq(a:float(), c:float())
   end
end

for _,name in ipairs({'lt','le','gt','ge','ne','eq'}) do
   cltorch.tests.tensor['self_' .. name] = function()
      c = torch.ClTensor{{4,   2,   -1},
                         {2, 5, 2}}
      d = torch.ClTensor{{3,   5,   -1},
                         {1,5, 3}}
      a = c:float()
      b = d:float()
      print(loadstring('c = c:' .. name .. '(d)')())
      print(loadstring('a = a:' .. name .. '(b)')())
      tester:asserteq(a:float(), c:float())
   end
end

for _,name in ipairs({'add', 'mul', 'div', 'pow',
         'lt', 'le', 'gt', 'ge', 'ne', 'eq'}) do
   cltorch.tests.tensor['outplace_' .. name] = function()
      c = torch.ClTensor{{4,   2,   -1},
                         {3.1,1.2, 4.9}}
      a = c:float()
      print(loadstring('c = torch.' .. name .. '(c, 3.4)')())
      print(loadstring('a = torch.' .. name .. '(a, 3.4)')())
      tester:asserteq(a:float(), c:float())
   end
end

for _,pair in ipairs({{'plus','+'},{'div', '/'}, {'mul', '*'},{'sub', '-'}}) do
   cltorch.tests.tensor['operator_' .. pair[1] .. '_scalar'] = function()
      c = torch.ClTensor{{4,   2,   -1},
                         {3.1,1.2, 4.9}}
      name = pair[2]
      a = c:float()
      print(loadstring('c = c ' .. name .. ' 3.4')())
      print(loadstring('a = a ' .. name .. ' 3.4')())
      tester:asserteq(a, c:float())
   end
end

for _, pair in ipairs({{'plus','+'}, {'sub','-'}}) do
   cltorch.tests.tensor['operator_' .. pair[1]] = function()
      c = torch.ClTensor{{4,   2,   -1},
                         {3.1,1.2, 4.9}}
      d = torch.ClTensor{{3,   5,   -2},
                         {2.1,2.2, 3.9}}
      a = c:float()
      b = d:float()
      op = pair[2]
      loadstring('a = a ' .. op .. ' b')()
      loadstring('c = c ' .. op .. ' d')()
      tester:asserteq(a, c:float())
   end
end

function cltorch.tests.tensor.test_perelement()
   c = torch.ClTensor{{4,   2,   -1},
                      {3.1,1.2, 4.9}}
   d = torch.ClTensor{{3,   5,   -2},
                      {2.1,2.2, 3.9}}
   a = c:float()
   b = d:float()
   c:add(d)
   a:add(b)
   tester:asserteq(a, c:float())

   c = torch.ClTensor{{4,   2,   -1},
                      {3.1,1.2, 4.9}}
   d = torch.ClTensor{{3,   5,   -2},
                      {2.1,2.2, 3.9}}
   a = c:float()
   b = d:float()
   a:cmul(b)
   c:cmul(d)
   tester:asserteq(a, c:float())


   c = torch.ClTensor{{4,   2,   -1},
                      {3.1,1.2, 4.9}}
   a = c:float()
   a = torch.clamp(a, 1.5, 4.5)
   c = torch.clamp(c, 1.5, 4.5)
   tester:asserteq(a, c:float())

   c = torch.ClTensor{{4,   2,   -1},
                      {3.1,1.2, 4.9}}
   a = c:float()
   c = -c
   a = -a
   tester:asserteq(a, c:float())
end

function cltorch.tests.tensor.test_blas()
   C = torch.ClTensor{{1,2,-1},
                      {3,4,0}}
   D = torch.ClTensor{{0,1},
                      {1,2},
                      {4,5}}
   A = C:float()
   B = D:float()
   A = torch.mm(A,B)
   C = torch.mm(C,D)
--   print('A\n', A)
--   print('C\n', C)
   tester:asserteq(A, C:float())

   C = torch.ClTensor{{1,2,-1},
                      {3,4,0}}
   D = torch.ClTensor{{0,1},
                      {1,2},
                      {4,5}}
   A = C:float()
   B = D:float()
   A = A * B
   C = C * D
--   print('A\n', A)
--   print('C\n', C)
   tester:asserteq(A, C:float())

   c = torch.ClTensor{3,5,1}
   d = torch.ClTensor{2,4,8}
   a = c:float()
   b = d:float()
   a = torch.dot(a,b)
   c = torch.dot(c,d)
   tester:asserteq(a, c)

   c = torch.ClTensor{3,5,1}
   d = torch.ClTensor{2,4,8}
   a = c:float()
   b = d:float()
   a = a * b
   c = c * d
   tester:asserteq(a, c)

   C = torch.ClTensor{{3,1,7},{3,2,4},{8,5,3}}
   d = torch.ClTensor{3,2,5}
   e = torch.ClTensor{3,1,2}
   
   tester:asserteq((torch.addr(C,d,e)):float(), torch.addr(C:float(), d:float(), e:float()))
end

function cltorch.tests.tensor.test_fills()
   C = torch.ClTensor(3,2)
   A = torch.FloatTensor(3,2)
   C:fill(1.345)
   A:fill(1.345)
   tester:asserteq(A, C:float())

   C = torch.ClTensor(3,2)
   A = torch.FloatTensor(3,2)
   C:fill(1.345)
   A:fill(1.345)
   C:zero()
   A:zero()
   tester:asserteq(A, C:float())
   
   A = torch.FloatTensor.zeros(torch.FloatTensor.new(), 3, 5)
   C = torch.ClTensor.zeros(torch.ClTensor.new(), 3, 5)
   tester:asserteq(A, C:float())

   A = torch.FloatTensor.ones(torch.FloatTensor.new(), 3, 5)
   C = torch.ClTensor.ones(torch.ClTensor.new(), 3, 5)
   tester:asserteq(A, C:float())

end

function cltorch.tests.tensor.test_matrixwide()
   C = torch.ClTensor{{3,2,4},{9,7,5}}
   A = C:float()
   tester:asserteq(A:max(), C:max())
   tester:asserteq(A:min(), C:min())
   tester:asserteq(A:sum(), C:sum())
   tester:asserteq(A:sum(1), C:sum(1):float())
   tester:asserteq(A:sum(2), C:sum(2):float())
end

function cltorch.tests.tensor.test_reshape()
   C = torch.ClTensor{{3,2,4},{9,7,5}}
   A = C:float()
   D = C:reshape(3,2)
   B = A:reshape(3,2)
   tester:asserteq(B, D:float())
end

function cltorch.tests.tensor.test_intpower()
   C = torch.ClTensor{{3.3,-2.2,0},{9,7,5}}
   Cpow = torch.pow(C,2)
   A = C:float()
   Apow = torch.pow(A,2)
   tester:asserteq(Apow, Cpow:float())
end

function cltorch.tests.tensor.test_powerofneg()
   C = torch.ClTensor{{3.3,-2.2,0},{9,7,5}}
   Cpow = torch.pow(C,2.4)
   A = C:float()
   Apow = torch.pow(A,2.4)
   tester:asserteq(Apow, Cpow:float())
end

function cltorch.tests.tensor.test_add()
   local s = torch.LongStorage{60,50}
   local A = torch.Tensor(s):uniform() - 0.5
   local B = torch.Tensor(s):uniform() - 0.5
   tester:asserteq(torch.add(A,B), torch.add(A:clone():cl(), B:clone():cl()):double())
   tester:asserteq(A + B, (A:clone():cl() + B:clone():cl() ):double())
   tester:asserteq(A:clone():add(B), (A:clone():cl():add(B:clone():cl())):double())
end

function cltorch.tests.tensor.test_cmul()
   local s = torch.LongStorage{60,50}
   local A = torch.Tensor(s):uniform() - 0.5
   local B = torch.Tensor(s):uniform() - 0.5
   tester:asserteq(torch.cmul(A,B), torch.cmul(A:clone():cl(), B:clone():cl()):double())
   tester:asserteq(A:clone():cmul(B), (A:clone():cl():cmul(B:clone():cl())):double())
end

function cltorch.tests.tensor.test_addcmul()
   local s = torch.LongStorage{60,50}
   local A = torch.Tensor(s):uniform() - 0.5
   local B = torch.Tensor(s):uniform() - 0.5
   local C = torch.Tensor(s):uniform() - 0.5
   tester:asserteq(torch.addcmul(A,1.234,B,C), torch.addcmul(A:clone():cl(), 1.234, B:clone():cl(), C:clone():cl()):double())
   tester:asserteq(A:clone():addcmul(1.234,B,C), (A:clone():cl():addcmul(1.234, B:clone():cl(),C:clone():cl())):double())
end

function cltorch.tests.tensor.test_addcdiv()
   torch.manualSeed(0)
   local s = torch.LongStorage{30,50}
   local A = torch.Tensor(s):uniform() - 0.5
   local B = torch.Tensor(s):uniform() - 0.5
   local C = torch.Tensor(s):uniform() - 0.5
   tester:asserteq(torch.addcdiv(A,1.234,B,C), torch.addcdiv(A:clone():cl(), 1.234, B:clone():cl(), C:clone():cl()):double())
   tester:asserteq(A:clone():addcdiv(1.234,B,C), (A:clone():cl():addcdiv(1.234, B:clone():cl(),C:clone():cl())):double())
end

-- this function doesnt exist in base torch
function cltorch.tests.tensor.test_neg()
   -- no neg for Tensors, only for clTensor, but we can use '-' to 
   -- compare
   local s = torch.LongStorage{60,50}
   local A = torch.Tensor(s):uniform() - 0.5
   local negA = - A:clone()
   local negAcl1 = - A:clone():cl()
   local negAcl2 = A:clone():cl():neg()
   tester:asserteq(negA, negAcl1:double())
   tester:asserteq(negA, negAcl2:double())
end

-- this function doesnt exist in base torch
function cltorch.tests.tensor.test_sub()
   local s = torch.LongStorage{60,50}
   local A = torch.Tensor(s):uniform() - 0.5
   local B = torch.Tensor(s):uniform() - 0.5
   AsubB = A - B
   AsubBcl = A:clone():cl() - B:clone():cl()
   AsubBcl2 = A:clone():cl():csub(B:clone():cl())
--   AsubBcl3 = torch.sub(A:clone():cl(), B:clone():cl())
   tester:asserteq(AsubB, AsubBcl:double())
   tester:asserteq(AsubB, AsubBcl2:double())
--   tester:asserteq(AsubB, AsubBcl3:double())
end

function cltorch.tests.tensor.test_apply()
   local s = torch.LongStorage{6,4}
   local A = torch.Tensor(s):uniform() - 0.5
   local Aapply = A:clone():apply(function(x) return math.sqrt(x+3) + math.exp(x) end)
   local Aapplycl = A:clone():cl():apply("*out = sqrt(*out + 3) + exp(*out)")
   tester:asserteq(Aapply, Aapplycl:double())

   local Aapplycl_x = A:clone():cl():apply("x = sqrt(x + 3) + exp(x)")
   tester:asserteq(Aapply, Aapplycl_x:double())
end

function cltorch.tests.tensor.test_map()
   local s = torch.LongStorage{60,50}
   local A = torch.Tensor(s):uniform() - 0.5
   local B = torch.Tensor(s):uniform() - 0.5
   local AmapB = A:clone():map(B, function(x, y) return math.sqrt(x*x + y*y + 3 + math.exp(y)) end)
   local AmapBcl = A:clone():cl():map(B:clone():cl(), 
      "*out = sqrt(*out * *out + *in1 * *in1 + 3 + exp(*in1))")
   tester:asserteq(AmapB, AmapBcl:double())

   local Aapp2Bcl = A:clone():cl():apply2(B:clone():cl(), 
      "*out = sqrt(*out * *out + *in1 * *in1 + 3 + exp(*in1))")
   tester:asserteq(AmapB, Aapp2Bcl:double())

   local Aapp2Bcl_xy = A:clone():cl():apply2(B:clone():cl(), 
      "x = sqrt(x * x + y * y + 3 + exp(y))")
   tester:asserteq(AmapB, Aapp2Bcl_xy:double())
end

function cltorch.tests.tensor.test_map2()
   local s = torch.LongStorage{60,50}
   local A = torch.Tensor(s):uniform() - 0.5
   local B = torch.Tensor(s):uniform() - 0.5
   local C = torch.Tensor(s):uniform() - 0.5
   local Amap2BC = A:clone():map2(B, C, 
      function(x, y, z) return math.sqrt(x*x + y*y + z + 3 + math.exp(z)) end)
   local Amap2BCcl = A:clone():cl():map2(
      B:clone():cl(),
      C:clone():cl(), 
      "*out = sqrt(*out * *out + *in1 * *in1 + *in2 + 3 + exp(*in2))")

   local Aapp3BCcl = A:clone():cl():apply3(
      B:clone():cl(),
      C:clone():cl(), 
      "*out = sqrt(*out * *out + *in1 * *in1 + *in2 + 3 + exp(*in2))")
   tester:asserteq(Amap2BC, Aapp3BCcl:double())

   local Aapp3BCcl_xyz = A:clone():cl():apply3(
      B:clone():cl(),
      C:clone():cl(), 
      "x = sqrt(x * x + y * y + z + 3 + exp(z))")
   tester:asserteq(Amap2BC, Aapp3BCcl_xyz:double())
end

function cltorch.tests.tensor.test_reduceAll()
   -- test on a large tensor, that needs two-pass :-)
   A = torch.Tensor(28*28*1280,10):uniform() - 0.5
   Asum = torch.sum(A)
   Aclsum = torch.sum(A:clone():cl())
   diff = Asum - Aclsum
   if diff < 0 then
      diff = - diff
   end
   tester:assert(diff < 1.2)

   -- now test on a single pass
   A = torch.Tensor(50,40):uniform() - 0.5
   Asum = torch.sum(A)
   Aclsum = torch.sum(A:clone():cl())
   diff = Asum - Aclsum
   if diff < 0 then
      diff = - diff
   end
   tester:assert(diff < 0.1)
end

function cltorch.tests.tensor.test_prodall()
   local s = torch.LongStorage{60,50}
   local A = torch.Tensor(s):uniform() - 0.5
   local Acl = A:cl()
--   print('allocated A,Acl')
   local Aprodall = torch.prod(A)
   local Aclprodall = torch.prod(Acl)
--   print('done prodall calcs')
   tester:asserteq(Aprodall, Aclprodall)

   Aprodall2 = A:prod()
--   print('calcing...')
   Aclprodall2 = Acl:prod()
--   print('...calced')
   tester:asserteq(Aprodall, Aprodall2)
   tester:asserteq(Aprodall, Aclprodall2)
end

function cltorch.tests.tensor.test_sumall()
   local s = torch.LongStorage{60,50}
   local A = torch.Tensor(s):uniform() - 0.5
   local Acl = A:cl()
--   print('allocated A,Acl')
   local Asumall = torch.sum(A)
   local Aclsumall = torch.sum(Acl)
--   print('done sumall calcs')
   tester:asserteq(torch.FloatTensor{Asumall}, torch.FloatTensor{Aclsumall})

   Aclsumall2 = Acl:sum()
   tester:asserteq(torch.FloatTensor{Asumall}, torch.FloatTensor{Aclsumall2})
end

function cltorch.tests.tensor.test_meanall()
   local s = torch.LongStorage{60,50}
   local A = torch.Tensor(s):uniform() - 0.5
   local Acl = A:cl()
   local res_cpu = torch.mean(A)
   local res_gpu = torch.mean(Acl)
   tester:asserteq(torch.FloatTensor{res_cpu}, torch.FloatTensor{res_gpu})

   res_gpu2 = Acl:mean()
   tester:asserteq(torch.FloatTensor{res_cpu}, torch.FloatTensor{res_gpu2})
end

function cltorch.tests.tensor.test_sumallt()
   local s = torch.LongStorage{60,50}
   local A = torch.Tensor(s):uniform() - 0.5
   local Acl = A:cl()
--   print('allocated A,Acl')
   local Asumall = torch.sum(A:t())
   local Aclsumall = torch.sum(Acl:t())
--   print('done sumall calcs')
   tester:asserteq(torch.FloatTensor{Asumall}, torch.FloatTensor{Aclsumall})

   Aclsumall2 = Acl:sum()
   tester:asserteq(torch.FloatTensor{Asumall}, torch.FloatTensor{Aclsumall2})
end

function cltorch.tests.tensor.test_prod()
   local s = torch.LongStorage{60,50}
   local A = torch.Tensor(s):uniform() - 0.5
   local Acl = A:cl()
   tester:asserteq(A:prod(1), Acl:prod(1):double())
   tester:asserteq(A:prod(2), Acl:prod(2):double())
end

function cltorch.tests.tensor.test_mean()
   local s = torch.LongStorage{60,50}
   local A = torch.Tensor(s):uniform() - 0.5
   local Acl = A:cl()
   tester:asserteq(A:mean(1), Acl:mean(1):double())
   tester:asserteq(A:mean(2), Acl:mean(2):double())
end

function cltorch.tests.tensor.test_sum()
   local s = torch.LongStorage{60,50}
   local A = torch.Tensor(s):uniform() - 0.5
   local Acl = A:cl()
   tester:asserteq(A:sum(1), Acl:sum(1):double())
   tester:asserteq(A:sum(2), Acl:sum(2):double())
end

function cltorch.tests.tensor.test_sum_t()
   local s = torch.LongStorage{60,50}
   local A = torch.Tensor(s):uniform() - 0.5
   local Acl = A:cl()
   tester:asserteq(A:t():sum(1), Acl:t():sum(1):double())
   tester:asserteq(A:t():sum(2), Acl:t():sum(2):double())
end

function cltorch.tests.tensor.test_sum_t_offset()
   local s = torch.LongStorage{60,50}
   local A = torch.Tensor(s):uniform() - 0.5
   local Acl = A:cl()
   tester:asserteq(A:narrow(1,10,30):t():sum(1), Acl:narrow(1,10,30):t():sum(1):double())
   tester:asserteq(A:narrow(1,10,30):t():sum(2), Acl:narrow(1,10,30):t():sum(2):double())
   tester:asserteq(A:narrow(2,10,30):t():sum(1), Acl:narrow(2,10,30):t():sum(1):double())
   tester:asserteq(A:narrow(2,10,30):t():sum(2), Acl:narrow(2,10,30):t():sum(2):double())
end

--function cltorch.tests.tensor.test_sum_t_offset()
--   local s = torch.LongStorage{60,50}
--   local A = torch.Tensor(s):uniform() - 0.5
----   A = A:narrow(2, 10, 30)
--   local Acl = A:cl()
--   tester:asserteq(A:t():sum(1), Acl:t():sum(1):double())   
--   tester:asserteq(A:t():sum(2), Acl:t():sum(2):double())   
--end

function cltorch.tests.tensor.test_max1()
   local s = torch.LongStorage{5,2}
   local A = torch.Tensor(s):uniform() - 0.5
   local Acl = A:cl()
   local Amax, Aind = A:max(1)
   local Aclmax, Aclind = Acl:max(1)
--   print('A max', Amax, Aind)
--   print('Acl max', Aclmax, Aclind)
   tester:asserteq(Amax, Aclmax:double())
   tester:asserteq(Aind, Aclind:long())
end

function cltorch.tests.tensor.test_max2()
   local s = torch.LongStorage{5,2}
   local A = torch.Tensor(s):uniform() - 0.5
   local Acl = A:cl()
   local Amax, Aind = A:max(2)
   local Aclmax, Aclind = Acl:max(2)
--   print('A max', Amax, Aind)
--   print('Acl max', Aclmax, Aclind)
   tester:asserteq(Amax, Aclmax:double())
   tester:asserteq(Aind, Aclind:long())
end

function cltorch.tests.tensor.test_min1()
   local s = torch.LongStorage{5,2}
   local A = torch.Tensor(s):uniform() - 0.5
   local Acl = A:cl()
   local Amax, Aind = A:min(1)
   local Aclmax, Aclind = Acl:min(1)
--   print('A max', Amax, Aind)
--   print('Acl max', Aclmax, Aclind)
   tester:asserteq(Amax, Aclmax:double())
   tester:asserteq(Aind, Aclind:long())
end

function cltorch.tests.tensor.test_min2()
   local s = torch.LongStorage{5,2}
   local A = torch.Tensor(s):uniform() - 0.5
   local Acl = A:cl()
   local Amax, Aind = A:min(2)
   local Aclmax, Aclind = Acl:min(2)
--   print('A max', Amax, Aind)
--   print('Acl max', Aclmax, Aclind)
   tester:asserteq(Amax, Aclmax:double())
   tester:asserteq(Aind, Aclind:long())
end

function cltorch.tests.tensor.test_indexfill()
   x = torch.Tensor(60,50):uniform() - 0.5
   selector = torch.LongTensor{4,7,12,19,35}
   value = -7
   xfill = x:clone()
   xfill:indexFill(2, selector, value)
--   print('xfill\n', xfill)

   xfillcl = x:clone():cl()
   xfillcl:indexFill(2, selector, value)
--   print('xfillcl\n', xfillcl)

   tester:asserteq(xfill, xfillcl:double())
end

function cltorch.tests.tensor.test_indexcopy()
   x = torch.Tensor(60,50):uniform() - 0.5
   z = torch.Tensor(60,2)
   z:select(2,1):fill(-1)
   z:select(2,2):fill(-2)
   xafter = x:clone()
   xafter:indexCopy(2,torch.LongTensor{5,1}, z)

   xaftercl = x:cl()
   xaftercl:indexCopy(2,torch.LongTensor{5,1}, z:cl())
   tester:asserteq(xafter, xaftercl:double())
end

function cltorch.tests.tensor.test_indexselect()
   local x = torch.Tensor(60,50):uniform() - 0.5

   xcopy = x:clone()
   xcopy:select(1, 2):fill(2) -- select row 2 and fill up
   xcopy:select(2, 5):fill(5) -- select column 5 and fill up

   xcopycl = x:cl()
   xcopycl:select(1, 2):fill(2) -- select row 2 and fill up
   xcopycl:select(2, 5):fill(5) -- select column 5 and fill up
   
   tester:asserteq(xcopy, xcopycl:double())
end

function cltorch.tests.tensor.test_cumsum()
   x = torch.Tensor(50,60):uniform() - 0.5
   tester:asserteq(torch.cumsum(x), torch.cumsum(x:cl()):double())
   tester:asserteq(torch.cumsum(x, 1), torch.cumsum(x:cl(), 1):double())
   tester:asserteq(torch.cumsum(x, 2), torch.cumsum(x:cl(), 2):double())
end

function cltorch.tests.tensor.test_cumprod()
   x = torch.Tensor(50,60):uniform() - 0.5
   tester:asserteq(torch.cumprod(x), torch.cumprod(x:cl()):double())
   tester:asserteq(torch.cumprod(x, 1), torch.cumprod(x:cl(), 1):double())
   tester:asserteq(torch.cumprod(x, 2), torch.cumprod(x:cl(), 2):double())
end

function cltorch.tests.tensor.test_gather()
   a = torch.Tensor(60,4)
   a:copy(torch.range(1,a:nElement()))
   acl = a:clone():cl()
   idx = torch.LongTensor({{2,1,3,1}})
   idxcl = idx:clone():cl()

   tester:asserteq(a:gather(1, idx), acl:gather(1, idxcl):double())
   tester:asserteq(torch.gather(a, 1, idx), torch.gather(acl, 1, idxcl):double())
end

function cltorch.tests.tensor.test_gather_t()
   a = torch.Tensor(60,4)
   a:copy(torch.range(1,a:nElement()))
   acl = a:clone():cl()

   idx = torch.LongTensor({{2,1,3,1}})
   idxcl = idx:clone():cl()

   a = a:t()
   acl = acl:t()
   idx = idx:t()
   idxcl = idxcl:t()

   tester:asserteq(a:gather(2, idx), acl:gather(2, idxcl):double())
   tester:asserteq(torch.gather(a, 2, idx), torch.gather(acl, 2, idxcl):double())
end

function cltorch.tests.tensor.test_gather_narrowed()
   a = torch.Tensor(60,4)
   a:copy(torch.range(1,a:nElement()))
   acl = a:clone():cl()

   idx = torch.LongTensor({{2,1,3,1}})
   idxcl = idx:clone():cl()

   a = a:narrow(1, 20, 20)
   acl = acl:narrow(1, 20, 20)

   cltorch.setTrace(1)
   tester:asserteq(a:gather(1, idx), acl:gather(1, idxcl):double())
   tester:asserteq(torch.gather(a, 1, idx), torch.gather(acl, 1, idxcl):double())
   cltorch.setTrace(0)
end

function cltorch.tests.tensor.test_scatter()
   x = torch.Tensor(2,5)
   x = x:copy(torch.range(1, x:nElement()))
   y = torch.zeros(3,5)
   idx = torch.LongTensor{{1,2,3,1,1},{3,1,1,2,3}}

   z = y:clone():scatter(1, idx:clone(), x:clone())
   zcl = y:clone():cl():scatter(1, idx:clone():cl(), x:clone():cl())

   tester:asserteq(z, zcl:double())
end

function cltorch.tests.tensor.test_scatterFill()
   y = torch.zeros(3,5)
   idx = torch.LongTensor{{1,2,3,1,1},{3,1,1,2,3}}

   z = y:clone():scatter(1, idx:clone(), 3.4567)
   zcl = y:clone():cl():scatter(1, idx:clone():cl(), 3.4567)

   tester:asserteq(z, zcl:double())
end

function cltorch.tests.tensor.test_cmin()
   local a = torch.Tensor(5,6):uniform() - 0.5
   local b = torch.Tensor(5,6):uniform() - 0.5
--   local res = a:cmin(b)
   local rescl = a:cl():cmin(b:cl())
--   tester:asserteq(res, rescl:double())
end

function cltorch.tests.tensor.test_save()
   a = torch.ClTensor{3,5,4.7, 0/0, 1/0, nil}
--   print('a', a)

   torch.save('out.dat~', a)

   b = torch.load('out.dat~')
--   print('b', b)
   tester:asserteq(torch.type(a), torch.type(b))
   af = a:float()
   bf = b:float()
   tester:asserteq(a:size(), b:size())
   for i=1,5 do
      if af[i] == af[i] then
         tester:asserteq(af[i], bf[i])
      else
         tester:assertne(bf[i], bf[i])
      end
   end
end

function cltorch.tests.tensor.test_addcdivshape()
   a = torch.ClTensor(3,2,4):uniform()
   b = torch.ClTensor(3*2*4):uniform()
   tester:asserteq(3, a:dim())
   a:addcdiv(1, b, b)
   tester:asserteq(3, a:dim())
end

function cltorch.tests.tensor.test_get()
   -- have to support it, because lbfgs uses it
   -- a bit slow though...
   acl = torch.ClTensor(20,5,14)
   acl[15][2][12] = 250
   acl[15][2][14] = 255
--   acl = a:cl()
   tester:asserteq('torch.ClTensor', torch.type(acl))
   tester:asserteq(250, acl[15][2][12])
   tester:asserteq(255, acl[15][2][14])
end

local function setUp()
   --cltorch.setDevice(1)
   print('')
end

local test = {}
for k,v in pairs(cltorch.tests.tensor) do
   test[k] = function()
      setUp()
      v()
   end
end

function cltorch.tests.tensor.test()
   tester = torch.Tester()
   tester:add(test)
   tester:run(tests)
   return #tester.errors == 0
end

if runtests then
   return cltorch.tests.tensor.test()
end


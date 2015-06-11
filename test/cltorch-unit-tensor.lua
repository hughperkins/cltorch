luaunit = require('luaunit')
require 'cltorch'

function test_basic()
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
end

os.exit( luaunit.LuaUnit.run() )



luaunit = require('luaunit')
require 'cltorch'

function test_basic()
  c = torch.ClStorage()
  s = '' .. c
  print('s', s)

--  c = torch.ClStorage(3)
--  print('c2\n', c)
--  c[1] = 5
--  print('c3\n', c)
--  c[3] = 7
--  print('c4\n', c)
--  -- print('c' .. c)

--  c = torch.ClStorage{4,9,2}
--  print('c5\n', c)

--  c:fill(7)
--  print('c6\n', c)

--  a = torch.Storage{1.5, 2.4, 5.3}
--  print('a\n', a)

--  b = torch.Storage(3)
--  b:copy(a)
--  print('bbbbb\n', b)

--  c:copy(a)
--  print('c7\n', c)

--  c[2] = 21
--  print('c8\n', c)
--  print('aaaaaa\n', a)
--  a:copy(c)
--  print('aaaaaa\n', a)

--  d = torch.ClStorage(3)
--  d:copy(c)
--  print('dddd\n', d)
end

os.exit( luaunit.LuaUnit.run() )


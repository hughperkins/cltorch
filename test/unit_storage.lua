-- local luaunit = require('cltorch.luaunit')

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

cltorch.tests.storage = {}

local function assertStrContains(target, value )
  local res = string.find(target, value)
  if res == nil then
    print('assertStrContains fail: [' .. string.gsub(target, '\n', '\\n\n') .. '] not contains [' .. string.gsub(value, '\n', '\\n\n') .. ']')
    tester:assert(string.find(target, value) ~= nil)
  end
end

function cltorch.tests.storage.test_basic()
  tester:asserteq('\n[torch.ClStorage of size 0]\n', tostring(torch.ClStorage()))
  assertStrContains(tostring(torch.ClStorage(3)), '\n%[torch.ClStorage of size 3%]\n')
  tester:asserteq(tostring(torch.ClStorage{4,9,2}), '\n 4\n 9\n 2\n[torch.ClStorage of size 3]\n')
  tester:asserteq(tostring(torch.ClStorage{1.5,2.4,5.3}), '\n 1.5000\n 2.4000\n 5.3000\n[torch.ClStorage of size 3]\n')

  c = torch.ClStorage{4,9,2}
  c:fill(7)
  tester:asserteq(tostring(c), '\n 7\n 7\n 7\n[torch.ClStorage of size 3]\n')

  c = torch.ClStorage{4,9,2}
  c:copy(torch.Storage{1.5,2.4,5.3})
  tester:asserteq(tostring(c), '\n 1.5000\n 2.4000\n 5.3000\n[torch.ClStorage of size 3]\n')

  a = torch.Storage(3)
  c = torch.ClStorage{4,9,2}
  a:copy(c)  
  tester:asserteq(tostring(a), '\n 4\n 9\n 2\n[torch.DoubleStorage of size 3]\n')

-- removed, since copies whole buffer :-(
--  c = torch.ClStorage{4,9,2}
--  c[2] = 21
--  tester:asserteq(tostring(c), '\n  4\n 21\n  2\n[torch.ClStorage of size 3]\n')

  c = torch.ClStorage{4,9,2}
  d = torch.ClStorage(3)
  d:copy(c)
  tester:asserteq(tostring(d), '\n 4\n 9\n 2\n[torch.ClStorage of size 3]\n')
  tester:asserteq(3, #d)
  tester:asserteq(3, d:size())

  c:resize(5)
  tester:asserteq(5, #c)
  c:fill(1)
  tester:asserteq(tostring(c), '\n 1\n 1\n 1\n 1\n 1\n[torch.ClStorage of size 5]\n')
end

local function setUp()
  cltorch.setDevice(1)
  print('')
end

local test = {}
for k,v in pairs(cltorch.tests.storage) do
  test[k] = function()
    setUp()
    v()
  end
end

function cltorch.tests.storage.test()
--  luaunit.LuaUnit.runSuite(cltorch.tests.storage)
   tester = torch.Tester()
   tester:add(test)
   tester:run(tests)
end

if runtests then
--  os.exit( luaunit.LuaUnit.run() )
  cltorch.tests.storage.test()
end


function cltorch.test()
   print('running tests...')
   luaunit = require('luaunit')

   include('test/cltorch-unit-storage.lua')
   test_basic()
   print('all tests finished')
end


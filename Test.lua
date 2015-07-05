function cltorch.test()
   print('running tests...')
   -- luaunit = require('luaunit')

   require('cltorch.unit_storage')
  print('aftter requiring cltorch.unit_storage')
   -- test_basic()
   cltorch.tests.storage.test()

   require('cltorch.unit_tensor')
  print('aftter requiring cltorch.unit_tensor')
   -- test_basic()
   cltorch.tests.tensor.test()

   print('all tests finished')
end


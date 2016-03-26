function cltorch.test()
   print('running tests...')
   -- luaunit = require('luaunit')

   require('cltorch.unit_storage')
   print('after requiring cltorch.unit_storage')
   -- test_basic()
   local res = cltorch.tests.storage.test()
   print('res', res)
   assert(res == true)

   require('cltorch.unit_tensor')
   print('aftter requiring cltorch.unit_tensor')
   -- test_basic()
   res = cltorch.tests.tensor.test()
   assert(res == true)

   print('all tests finished')
end


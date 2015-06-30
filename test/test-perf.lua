require 'cltorch'

function test_apply2(its)
  a = torch.ClTensor(50, 500)
  a:uniform()
  b = torch.ClTensor(50, 500)
  b:uniform()
  a:add(b)
  cltorch.dumpTimings()
  for it=1,its do
    a:add(b)    
  end
  cltorch.dumpTimings()
end


cltorch.setAddFinish(1)
test_apply2(10000)


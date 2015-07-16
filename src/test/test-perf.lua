require 'cltorch'

function test_apply1(its)
  a = torch.ClTensor(50, 500)
  a:uniform()
  a:add(1)
  cltorch.dumpProfiling()
  cltorch.dumpTimings()
  for it=1,its do
    a:add(it)    
  end
  cltorch.dumpTimings()
end

function test_apply2(its)
  a = torch.ClTensor(50, 500)
  a:uniform()
  b = torch.ClTensor(50, 500)
  b:uniform()
  a:add(b)
  cltorch.dumpProfiling()
  cltorch.dumpTimings()
  for it=1,its do
    a:add(b)    
  end
  cltorch.dumpTimings()
end

function test_scatterFill(its)
  idx = torch.multinomial(torch.range(1,10):reshape(10,1):expand(10,10):t(),10):t():cl()
  a = torch.Tensor(10,10):uniform():mul(100):int():cl()
  c = a:scatter(1,idx,3)
  cltorch.dumpTimings()
  for it=1,its do
    a:scatter(1,idx,it)
  end
  cltorch.dumpTimings()
end

--cltorch.setAddFinish(1)
cltorch.setDevice(1)
cltorch.setProfiling(1)
test_apply1(500)
--test_apply2(500)
-- test_scatterFill(10000)
cltorch.dumpProfiling()



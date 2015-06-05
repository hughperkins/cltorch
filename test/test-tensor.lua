print("running require clnn...")
require 'clnn'
print("... require clnn done")

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

-- d:copy(c)
-- print('d2', d)


require 'cltorch'

a = torch.ClTensor{3,5,4.7, 0/0, 1/0, nil}
print('a', a)

torch.save('out.dat~', a)

b = torch.load('out.dat~')
print('b', b)


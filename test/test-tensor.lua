print("running require clnn...")
require 'clnn'
print("... require clnn done")

a = torch.Tensor{3,5,2}
print('a\n', a)

c = torch.ClTensor{7,4,5}


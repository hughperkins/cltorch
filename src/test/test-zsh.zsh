
ps
source ~/torch/install/bin/torch-activate
env | grep LD
env | grep PATH
env | grep LUA
luajit -e 'print("hello")'
luajit -l torch -e 'print(torch.Tensor(3,2):uniform())'
luajit -l cltorch -e 'print(torch.ClTensor(3,2):uniform())'


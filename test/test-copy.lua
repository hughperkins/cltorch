require 'cutorch'

--local torchtypes = {}
--table.insert(torchtypes, torch.DoubleTensor)
--table.insert(torchtypes, torch.FloatTensor)
--table.insert(torchtypes, torch.IntTensor)
--table.insert(torchtypes, torch.ByteTensor)
--table.insert(torchtypes, torch.CharTensor)
--table.insert(torchtypes, torch.ShortTensor)
--table.insert(torchtypes, torch.LongTensor)

--for i,torchtype in ipairs(torchtypes) do
--  torchtype.oldcopy = torchtype.copy
--end

require 'cltorch'

--for i,torchtype in ipairs(torchtypes) do
--  torchtype.newcopy = torchtype.copy
--end

--for i,torchtype in ipairs(torchtypes) do
--  function torchtype.copy(self, two)
--    for j, innertorchtype in ipairs(torchtypes) do
--      if(torch.type(two) == getmetatable(innertorchtype).__typename) then
--        torchtype.oldcopy(self, two)
--        return self
--      end
--    end

--    if(torch.type(two) == "torch.ClTensor") then
--      torchtype.newcopy(self, two)
--      return self
--    end

--    torchtype.oldcopy(self, two)
--    return self
--  end
--end

a = torch.Tensor{3,5,2}
print('a\n', a)

b = torch.Tensor{3,5,2}:cl()
bd = torch.DoubleTensor()
print('bd a\n', bd)
bd:resize(b:size())
print('bd b\n', bd)
bd:copy(b)
print('bd c\n', bd)
--local tensor = torch.DoubleTensor():resize(self:size()):copy(self)

print('b\n', b)

c = torch.Tensor{3,5,2}:cuda()
print('c\n', c)

print(torch.FloatTensor{4,5,6}:cuda())

print(torch.FloatTensor{4,5,6}:cuda():float())
print(torch.FloatTensor{4,5,6}:cuda():int())
print(torch.FloatTensor{4,5,6}:cuda():byte())

print(torch.FloatTensor{4,5,6}:cl():float())
print(torch.FloatTensor{4,5,6}:cl():int())
print(torch.FloatTensor{4,5,6}:cl():byte())


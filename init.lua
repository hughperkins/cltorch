require "torch"

-- store old copy functions, in case cutorch has been loaded
-- note that this only works if cutorch is loaded first

local torchtypes = {}
table.insert(torchtypes, torch.DoubleTensor)
table.insert(torchtypes, torch.FloatTensor)
table.insert(torchtypes, torch.IntTensor)
table.insert(torchtypes, torch.ByteTensor)
table.insert(torchtypes, torch.CharTensor)
table.insert(torchtypes, torch.ShortTensor)
table.insert(torchtypes, torch.LongTensor)

for i,torchtype in ipairs(torchtypes) do
  torchtype.cloldcopy = torchtype.copy
end

cltorch = paths.require("libcltorch")

for i,torchtype in ipairs(torchtypes) do
  torchtype.clnewcopy = torchtype.copy
end

for i,torchtype in ipairs(torchtypes) do
--  print('init torch type copy', torchtype)
  torchtype.copy = function (self, two)
--    print('cust copy')
    for j, innertorchtype in ipairs(torchtypes) do
      if(torch.type(two) == getmetatable(innertorchtype).__typename) then
        torchtype.cloldcopy(self, two)
        return self
      end
    end

    if(torch.type(two) == "torch.ClTensor") then
      torchtype.clnewcopy(self, two)
      return self
    end

    torchtype.cloldcopy(self, two)
    return self
  end
end

-- convert to FloatStorage first, rather than repeatedly
-- calling 'get' on ClStorage
function torch.ClStorage.__tostring__(self)
  floatstorage = torch.FloatStorage(self:size())
  floatstorage:copy(self)
  return string.gsub(floatstorage:__tostring__(), 'FloatStorage', 'ClStorage')
end

--torch.ClStorage.__tostring__ = torch.FloatStorage.__tostring__
torch.ClTensor.__tostring__ = torch.FloatTensor.__tostring__

include('Tensor.lua')
--include('FFI.lua')
--include('test.lua')

--local unpack = unpack or table.unpack

return cltorch


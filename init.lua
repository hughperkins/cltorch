require "torch"
cltorch = paths.require("libcltorch")

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


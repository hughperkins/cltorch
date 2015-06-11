require "torch"
cltorch = paths.require("libcltorch")

torch.ClStorage.__tostring__ = torch.FloatStorage.__tostring__
torch.ClTensor.__tostring__ = torch.FloatTensor.__tostring__

include('Tensor.lua')
--include('FFI.lua')
--include('test.lua')

--local unpack = unpack or table.unpack

--local function Module__cl(self)
--   print("Module__cl")
--end

--require 'nn'

--rawset(torch.getmetatable('nn.Module'), 'cl', Module__cl)

return cltorch


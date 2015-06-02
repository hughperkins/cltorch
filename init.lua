require "torch"
clnn = paths.require("libclnn")

local function Module__cl(self)
   print("Module__cl")
end

require 'nn'

rawset(torch.getmetatable('nn.Module'), 'cl', Module__cl)


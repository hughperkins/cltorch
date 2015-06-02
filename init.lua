-- Copyright Hugh Perkins 2015 hughperkins at gmail
--
-- This Source Code Form is subject to the terms of the Mozilla Public License, 
-- v. 2.0. If a copy of the MPL was not distributed with this file, You can 
-- obtain one at http://mozilla.org/MPL/2.0/.

require "torch"
clnn = paths.require("libclnn")

local function Module__cl(self)
   print("Module__cl")
end

require 'nn'

rawset(torch.getmetatable('nn.Module'), 'cl', Module__cl)

